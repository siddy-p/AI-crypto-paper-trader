import logging
import datetime
import sqlite3
from flask import Flask, jsonify, render_template
from src import config
from src import database

logger = logging.getLogger(__name__)

def create_app(trader_bot):
    app = Flask(__name__, template_folder="templates")
    
    # Disable Flask default startup logs to keep console clean
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/status")
    def api_status():
        try:
            metrics = database.calculate_metrics()
            open_positions = database.get_open_trades()
            recent_trades = database.get_trade_history(limit=50)
            model_predictions = trader_bot.get_live_metrics_for_dashboard()
            
            # Enrich open positions with current market prices and P&L
            enriched_positions = []
            for pos in open_positions:
                symbol = pos["symbol"]
                current_price = trader_bot.latest_prices.get(symbol, pos["entry_price"])
                
                # Check position side for correct P&L direction
                side = pos.get("side", "BUY")
                if side == "SELL":
                    unrealized_pnl = (pos["entry_price"] - current_price) * pos["quantity"]
                    pnl_pct = ((pos["entry_price"] - current_price) / pos["entry_price"]) * 100.0
                else:
                    unrealized_pnl = (current_price - pos["entry_price"]) * pos["quantity"]
                    pnl_pct = ((current_price - pos["entry_price"]) / pos["entry_price"]) * 100.0
                
                enriched_positions.append({
                    "id": pos["id"],
                    "symbol": symbol,
                    "side": side,
                    "entry_price": pos["entry_price"],
                    "current_price": current_price,
                    "quantity": pos["quantity"],
                    "confidence": pos["confidence"],
                    "stop_loss": pos["stop_loss"],
                    "take_profit": pos["take_profit"],
                    "pnl": round(unrealized_pnl, 4),
                    "pnl_pct": round(pnl_pct, 2)
                })
            
            # Fetch database record counts to show update health
            conn = database.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM candles")
            candle_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM trades")
            trade_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM portfolio_history")
            history_count = cursor.fetchone()[0]
            conn.close()
            
            return jsonify({
                "success": True,
                "metrics": metrics,
                "open_positions": enriched_positions,
                "recent_trades": recent_trades,
                "model_predictions": model_predictions,
                "safety_disabled": config.DISABLE_ORDER_EXECUTION,
                "last_retrain_date": str(trader_bot.last_retrain_date) if trader_bot.last_retrain_date else "Never",
                "system_status": "RUNNING" if trader_bot.is_running else "STOPPED",
                "db_stats": {
                    "candles": candle_count,
                    "trades": trade_count,
                    "history": history_count
                }
            })
        except Exception as e:
            logger.error(f"Error serving status API: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/test_trade", methods=["POST"])
    def api_test_trade():
        try:
            # Force open a simulated trade for BTCUSDT at the latest streamed price
            symbol = "BTCUSDT"
            latest_price = trader_bot.latest_prices.get(symbol, 65000.0)
            
            # Check if already holding
            open_trades = database.get_open_trades()
            if any(t["symbol"] == symbol for t in open_trades):
                return jsonify({"success": False, "error": f"Already holding an open position for {symbol}."})
            
            # Alternate sides based on total database trade rows
            conn = database.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades")
            count = cursor.fetchone()[0]
            conn.close()
            
            side = "SELL" if count % 2 == 1 else "BUY"
            
            # Use a $10 position size for this test order
            usdt_size = 10.0
            qty = usdt_size / latest_price
            
            if side == "SELL":
                # For Short: Stop loss is above, take profit is below
                stop_loss = latest_price * 1.01
                take_profit = latest_price * 0.98
            else:
                # For Long: Stop loss is below, take profit is above
                stop_loss = latest_price * 0.99
                take_profit = latest_price * 1.02
            
            # Force simulated balance execution
            with trader_bot.lock:
                trader_bot.simulated_usdt -= usdt_size
                
            database.add_trade(
                symbol=symbol,
                side=side,
                entry_price=latest_price,
                quantity=qty,
                confidence=85.0,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Write new history snapshot
            portfolio_val = trader_bot.get_portfolio_value()
            available_cash = trader_bot.get_available_usdt()
            database.log_portfolio(portfolio_val, available_cash)
            
            logger.info(f"Triggered manual test trade on {symbol} ({side}) at price ${latest_price:.2f}")
            return jsonify({"success": True, "message": f"Successfully forced a test {side} on {symbol} at ${latest_price:.2f}!"})
        except Exception as e:
            logger.error(f"Failed to open manual test trade: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/portfolio_history")
    def api_portfolio_history():
        try:
            history = database.get_portfolio_history(limit=100)
            formatted = []
            for h in history:
                # Format ISO timestamp to more readable string
                dt_str = h["timestamp"]
                try:
                    dt = datetime.datetime.fromisoformat(dt_str)
                    label = dt.strftime("%m-%d %H:%M")
                except:
                    label = dt_str
                    
                formatted.append({
                    "label": label,
                    "value": h["total_value"]
                })
            return jsonify(formatted)
        except Exception as e:
            logger.error(f"Error serving portfolio history API: {e}")
            return jsonify([])

    @app.route("/api/db_inspect")
    def api_db_inspect():
        try:
            conn = database.get_db_connection()
            if database.DB_TYPE == "postgres":
                cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                
                cursor.execute("SELECT * FROM candles ORDER BY open_time DESC LIMIT 10")
                candles = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute("SELECT * FROM trades ORDER BY id DESC LIMIT 20")
                trades = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute("SELECT * FROM portfolio_history ORDER BY id DESC LIMIT 20")
                history = [dict(row) for row in cursor.fetchall()]
            else:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM candles ORDER BY open_time DESC LIMIT 10")
                candles = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute("SELECT * FROM trades ORDER BY id DESC LIMIT 20")
                trades = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute("SELECT * FROM portfolio_history ORDER BY id DESC LIMIT 20")
                history = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            return jsonify({
                "success": True,
                "candles": candles,
                "trades": trades,
                "portfolio_history": history
            })
        except Exception as e:
            logger.error(f"Error executing db_inspect: {e}")
            return jsonify({"success": False, "error": str(e)}), 500
     
    return app

def run_dashboard(app, host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT):
    logger.info(f"Starting web dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)
