import time
import logging
import threading
import datetime
import pandas as pd
from src import config
from src import database
from src.binance_client import BinanceClient
from src.model import CryptoModelManager
from src.indicators import calculate_all_indicators

logger = logging.getLogger(__name__)

class Trader:
    def __init__(self, client: BinanceClient, model_manager: CryptoModelManager):
        self.client = client
        self.model_manager = model_manager
        self.symbols = []
        self.buffers = {}       # Dict mapping symbol -> list of candle dicts
        self.latest_prices = {} # Dict mapping symbol -> latest price (float)
        self.lock = threading.Lock()
        
        # State variables
        self.is_running = False
        self.last_retrain_date = None
        self.simulated_usdt = config.INITIAL_USDT_BALANCE
        
        # Core settings
        self.risk_pct = 0.01      # 1% risk per trade
        self.stop_loss_pct = 0.01 # 1% SL
        self.take_profit_pct = 0.02 # 2% TP
        self.confidence_threshold = 75.0 # Min confidence 75%
        
        # Retraining scheduler state
        self.last_retrain_time = 0.0

    def bootstrap_system(self):
        """
        Bootstrapping routine:
        1. Query top 20 symbols.
        2. Fetch last 500 candles for each from Binance to seed buffers.
        3. Insert candles into SQLite cache.
        4. Train model if not already loaded.
        5. Initialize portfolio history.
        """
        logger.info("Initializing trading bot bootstrap...")
        database.init_db()
        
        # 1. Fetch top 20 symbols
        self.symbols = self.client.get_top_20_usdt_pairs()
        
        # 2. Seed buffers and database cache
        candles_to_save = []
        for symbol in self.symbols:
            logger.info(f"Bootstrapping historical data for {symbol}...")
            # Check latest cached candle time to see if we can perform an incremental fetch
            latest_time = database.get_latest_candle_time(symbol)
            
            # Fetch 500 candles from Binance (approximately 8 hours of 1m data)
            # If database is empty, fetch 1000 candles to give XGBoost plenty of samples
            fetch_limit = 1000 if latest_time == 0 else 500
            klines = self.client.fetch_historical_klines(symbol, limit=fetch_limit)
            
            for k in klines:
                candles_to_save.append((
                    symbol, k["open_time"], k["open"], k["high"], k["low"], k["close"], k["volume"]
                ))
            
            # Populate in-memory buffers
            self.buffers[symbol] = klines
            if klines:
                self.latest_prices[symbol] = klines[-1]["close"]
                
            time.sleep(0.1) # Small rate-limit protection

        # Save bootstrap candles to database
        if candles_to_save:
            database.save_candles(candles_to_save)
            logger.info(f"Saved {len(candles_to_save)} bootstrapped candles to SQLite.")

        # 3. Train or load the model
        if self.model_manager.model is None:
            logger.info("No model loaded. Starting initial model training...")
            success = self.model_manager.train(self.symbols)
            if not success:
                logger.error("Initial model training failed. Will run with default neutral model predictions.")
            self.last_retrain_time = time.time()
            self.last_retrain_date = datetime.date.today()
        else:
            self.last_retrain_time = time.time()
            self.last_retrain_date = datetime.date.today()

        # 4. Set portfolio balance
        self.sync_portfolio_balance()
        logger.info("Bot bootstrap successfully completed.")

    def sync_portfolio_balance(self):
        """Synchronizes USDT balances from Binance or local SQLite database."""
        with self.lock:
            # Try to get latest portfolio record to restore simulated USDT cash balance
            history = database.get_portfolio_history(limit=1)
            if history:
                self.simulated_usdt = history[0]["available_usdt"]
                total_value = history[0]["total_value"]
                logger.info(f"Restored simulated balance: USDT cash = {self.simulated_usdt:.2f}, Portfolio value = {total_value:.2f}")
            else:
                self.simulated_usdt = config.INITIAL_USDT_BALANCE
                database.log_portfolio(self.simulated_usdt, self.simulated_usdt)
                logger.info(f"Initialized portfolio balance at {self.simulated_usdt:.2f} USDT")

    def get_portfolio_value(self):
        """Calculates current total portfolio value (cash + value of open positions)."""
        open_trades = database.get_open_trades()
        positions_value = 0.0
        
        for trade in open_trades:
            symbol = trade["symbol"]
            qty = trade["quantity"]
            # Get latest price of asset, falling back to entry price
            current_price = self.latest_prices.get(symbol, trade["entry_price"])
            positions_value += qty * current_price
            
        if config.DISABLE_ORDER_EXECUTION:
            return self.simulated_usdt + positions_value
        else:
            # Query actual balances from Binance Testnet
            balances = self.client.get_account_balance()
            usdt_cash = balances.get("USDT", 0.0)
            
            # Sum up open positions based on current prices
            total_equity = usdt_cash
            for asset, free_balance in balances.items():
                if asset != "USDT" and free_balance > 0.0:
                    symbol = f"{asset}USDT"
                    price = self.latest_prices.get(symbol, 0.0)
                    if price > 0.0:
                        total_equity += free_balance * price
            return total_equity

    def get_available_usdt(self):
        if config.DISABLE_ORDER_EXECUTION:
            return self.simulated_usdt
        else:
            balances = self.client.get_account_balance()
            return balances.get("USDT", 0.0)

    def handle_live_tick(self, symbol, is_closed, open_time, open_price, high_price, low_price, close_price, volume):
        """
        Receives real-time ticks from WebSockets.
        Runs every tick to update current prices and check open position SL/TP exits.
        Runs once per minute on candle close to recalculate indicators and execute trades.
        """
        # 1. Update latest price
        self.latest_prices[symbol] = close_price
        
        # 2. Check stop-loss / take-profit exits for any open positions of this symbol
        self.check_positions_exits(symbol, close_price)

        # 3. Process new minute candles
        if is_closed:
            logger.info(f"New closed 1m candle for {symbol}: Close={close_price}, Vol={volume}")
            
            # Save candle to SQLite cache
            database.save_candles([(symbol, open_time, open_price, high_price, low_price, close_price, volume)])
            
            # Update memory buffers
            with self.lock:
                if symbol not in self.buffers:
                    self.buffers[symbol] = []
                self.buffers[symbol].append({
                    "open_time": open_time,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume
                })
                
                # Keep buffer capped to last 150 candles to maintain performance
                if len(self.buffers[symbol]) > 150:
                    self.buffers[symbol] = self.buffers[symbol][-150:]

            # Perform strategy checks
            self.execute_strategy(symbol)
            
            # Log current portfolio value to history
            portfolio_val = self.get_portfolio_value()
            available_cash = self.get_available_usdt()
            database.log_portfolio(portfolio_val, available_cash)
            
            # Check if we need to retrain daily
            self.check_daily_retrain()

    def execute_strategy(self, symbol):
        """Runs indicator calculations, queries the model, and executes trading actions."""
        # 1. Build DataFrame
        with self.lock:
            candles = list(self.buffers.get(symbol, []))
            
        if len(candles) < 60:
            # Insufficient buffer to calculate indicators reliably
            return

        df = pd.DataFrame(candles)
        df = calculate_all_indicators(df)
        
        # 2. Query model prediction
        prob = self.model_manager.predict_probability(df)
        confidence = prob * 100.0
        
        logger.debug(f"Strategy evaluate for {symbol} - Close: {df['close'].iloc[-1]}, Model Confidence: {confidence:.2f}%")
        
        # Store latest confidence score in memory (can be queried by the dashboard)
        df_latest_close = df["close"].iloc[-1]
        
        # 3. Check for BUY Signal
        # Executed if confidence exceeds 75% and we don't already have an open position for this symbol
        if confidence > self.confidence_threshold:
            open_trades = database.get_open_trades()
            already_holding = any(t["symbol"] == symbol for t in open_trades)
            
            if not already_holding:
                self.open_long_position(symbol, df_latest_close, confidence)

    def open_long_position(self, symbol, entry_price, confidence):
        """Calculates size, executes BUY order, and logs position in SQLite."""
        portfolio_val = self.get_portfolio_value()
        available_usdt = self.get_available_usdt()
        
        # Position sizing: Risk 1% of portfolio value per trade.
        # Loss amount on stop loss hit: Risk Amount = Portfolio Value * 0.01.
        # Since Stop Loss is 1%, USDT position size = Risk Amount / 0.01 = Portfolio Value.
        usdt_position_size = portfolio_val * self.risk_pct / self.stop_loss_pct
        
        # Capping position size to 98% of available USDT cash (allowing 2% fee buffer)
        usdt_to_spend = min(usdt_position_size, available_usdt * 0.98)
        
        # Minimum trade size check (Binance Testnet minimum Spot order size is usually 10 USDT)
        if usdt_to_spend < 10.0:
            logger.warning(f"Position size {usdt_to_spend:.2f} USDT is below minimum (10.0 USDT). Skipping order for {symbol}.")
            return
            
        qty = usdt_to_spend / entry_price
        
        # Calculate stops
        stop_loss = entry_price * (1.0 - self.stop_loss_pct)
        take_profit = entry_price * (1.0 + self.take_profit_pct)
        
        logger.info(f"★★★ BUY SIGNAL for {symbol}! Confidence: {confidence:.2f}%. Size: {usdt_to_spend:.2f} USDT, Price: {entry_price}")

        if config.DISABLE_ORDER_EXECUTION:
            # Simulated execution
            with self.lock:
                self.simulated_usdt -= usdt_to_spend
                
            database.add_trade(
                symbol=symbol,
                side="BUY",
                entry_price=entry_price,
                quantity=qty,
                confidence=confidence,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            logger.info(f"Simulated position logged: bought {qty:.6f} {symbol} at {entry_price:.6f}")
        else:
            # Real Binance Testnet Spot execution
            order_result = self.client.place_market_buy(symbol, usdt_to_spend)
            if order_result:
                # Extract executed price and actual quantity filled
                fills = order_result.get("fills", [])
                if fills:
                    total_qty = sum(float(f["qty"]) for f in fills)
                    weighted_price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / total_qty
                else:
                    total_qty = float(order_result.get("executedQty", qty))
                    weighted_price = float(order_result.get("cummulativeQuoteQty", usdt_to_spend)) / total_qty
                
                # Recalculate stops based on actual execution prices
                stop_loss = weighted_price * (1.0 - self.stop_loss_pct)
                take_profit = weighted_price * (1.0 + self.take_profit_pct)
                
                database.add_trade(
                    symbol=symbol,
                    side="BUY",
                    entry_price=weighted_price,
                    quantity=total_qty,
                    confidence=confidence,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                logger.info(f"Live Testnet position opened: bought {total_qty:.6f} {symbol} at {weighted_price:.6f}")

    def check_positions_exits(self, symbol, current_price):
        """Checks if any open positions for the given symbol have reached Stop-Loss or Take-Profit thresholds."""
        open_trades = database.get_open_trades()
        symbol_trades = [t for t in open_trades if t["symbol"] == symbol]
        
        for trade in symbol_trades:
            trade_id = trade["id"]
            qty = trade["quantity"]
            entry_price = trade["entry_price"]
            stop_loss = trade["stop_loss"]
            take_profit = trade["take_profit"]
            
            # Check Stop Loss (<= 1%)
            if current_price <= stop_loss:
                pnl = (current_price - entry_price) * qty
                logger.info(f"▲ STOP LOSS HIT for {symbol} at {current_price} (Entry: {entry_price}, P&L: {pnl:.4f} USDT)")
                self.close_position(trade_id, symbol, qty, entry_price, current_price, pnl, "STOP_LOSS")
                
            # Check Take Profit (>= 2%)
            elif current_price >= take_profit:
                pnl = (current_price - entry_price) * qty
                logger.info(f"▼ TAKE PROFIT HIT for {symbol} at {current_price} (Entry: {entry_price}, P&L: {pnl:.4f} USDT)")
                self.close_position(trade_id, symbol, qty, entry_price, current_price, pnl, "TAKE_PROFIT")

    def close_position(self, trade_id, symbol, quantity, entry_price, exit_price, pnl, reason):
        """Closes a position, places a market SELL order on Binance (or simulates), and logs the result."""
        if config.DISABLE_ORDER_EXECUTION:
            # Simulated sell
            with self.lock:
                self.simulated_usdt += quantity * exit_price
                
            database.close_trade(trade_id, exit_price, pnl, reason)
            logger.info(f"Simulated position closed: sold {quantity:.6f} {symbol} at {exit_price:.6f} (P&L: {pnl:.4f})")
        else:
            # Real Binance Testnet Spot execution
            order_result = self.client.place_market_sell(symbol, quantity)
            if order_result:
                fills = order_result.get("fills", [])
                if fills:
                    actual_qty = sum(float(f["qty"]) for f in fills)
                    actual_exit_price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / actual_qty
                else:
                    actual_qty = float(order_result.get("executedQty", quantity))
                    actual_exit_price = float(order_result.get("cummulativeQuoteQty", quantity * exit_price)) / actual_qty
                
                actual_pnl = (actual_exit_price - entry_price) * actual_qty
                database.close_trade(trade_id, actual_exit_price, actual_pnl, reason)
                logger.info(f"Live Testnet position closed: sold {actual_qty:.6f} {symbol} at {actual_exit_price:.6f} (P&L: {actual_pnl:.4f})")

    def check_daily_retrain(self):
        """Checks if 24 hours have passed since the last retraining, and triggers the process."""
        current_time = time.time()
        # Trigger retraining if 24 hours (86400 seconds) have passed
        if current_time - self.last_retrain_time >= 86400:
            logger.info("Daily retraining timer triggered. Preparing retraining data...")
            threading.Thread(target=self.run_daily_retrain_job).start()

    def run_daily_retrain_job(self):
        """Background job to download the latest candles, insert them, and retrain the model."""
        try:
            logger.info("Downloading latest market data for model retraining...")
            candles_to_save = []
            
            # Fetch last 1440 candles (24 hours) for each of the monitored symbols
            for symbol in self.symbols:
                latest_time = database.get_latest_candle_time(symbol)
                # Fetch starting from the latest candle time we have
                klines = self.client.fetch_historical_klines(symbol, limit=1440, start_time=latest_time)
                for k in klines:
                    candles_to_save.append((
                        symbol, k["open_time"], k["open"], k["high"], k["low"], k["close"], k["volume"]
                    ))
                time.sleep(0.1)
                
            if candles_to_save:
                database.save_candles(candles_to_save)
                logger.info(f"Downloaded and cached {len(candles_to_save)} new candles for retraining.")
                
            # Run model training
            success = self.model_manager.train(self.symbols)
            if success:
                self.last_retrain_time = time.time()
                self.last_retrain_date = datetime.date.today()
                # Log daily performance report
                database.generate_daily_report(datetime.date.today().isoformat())
                logger.info("Daily model retraining successfully completed and performance report generated.")
            else:
                logger.error("Daily model retraining failed.")
        except Exception as e:
            logger.error(f"Failed to execute daily model retraining: {e}")

    def get_live_metrics_for_dashboard(self):
        """Returns indicators, latest prices, and current model predictions for the dashboard."""
        results = []
        
        for symbol in self.symbols:
            latest_price = self.latest_prices.get(symbol, 0.0)
            
            # Check what confidence score the model predicts right now
            confidence = 50.0
            volume_change = 0.0
            atr = 0.0
            
            with self.lock:
                candles = list(self.buffers.get(symbol, []))
                
            if len(candles) >= 60:
                df = pd.DataFrame(candles)
                df = calculate_all_indicators(df)
                prob = self.model_manager.predict_probability(df)
                confidence = prob * 100.0
                volume_change = df["volume_change_pct"].iloc[-1]
                atr = df["atr_14"].iloc[-1]
                
            results.append({
                "symbol": symbol,
                "price": round(latest_price, 6) if latest_price > 0 else "N/A",
                "volume_change": round(volume_change, 2),
                "atr": round(atr, 6),
                "confidence": round(confidence, 1)
            })
            
        return results
