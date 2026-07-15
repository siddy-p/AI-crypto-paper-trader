import hmac
import hashlib
import time
import json
import logging
import threading
import requests
import websocket
from src import config

logger = logging.getLogger(__name__)

class BinanceClient:
    def __init__(self):
        self.api_key = config.BINANCE_API_KEY
        self.api_secret = config.BINANCE_API_SECRET
        self.disable_execution = config.DISABLE_ORDER_EXECUTION
        self.ws = None
        self.ws_thread = None
        self.is_connected = False
        
        # Safe fallback list of top 20 USDT pairs
        self.fallback_pairs = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "SHIBUSDT", "AVAXUSDT", "DOTUSDT",
            "MATICUSDT", "LINKUSDT", "NEARUSDT", "LTCUSDT", "BCHUSDT",
            "TRXUSDT", "ETCUSDT", "XLMUSDT", "UNIUSDT", "FILUSDT"
        ]

    def _get_signature(self, query_string):
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _send_signed_request(self, method, endpoint, params=None):
        """Sends a signed HTTP request to the Binance Testnet API."""
        if not params:
            params = {}
            
        if self.disable_execution:
            logger.info(f"Signed request {method} {endpoint} bypassed (DISABLE_ORDER_EXECUTION is True)")
            return {"mocked": True}
            
        if not self.api_key or not self.api_secret:
            raise ValueError("Binance Testnet API Key and Secret must be configured when DISABLE_ORDER_EXECUTION is False.")

        params["timestamp"] = int(time.time() * 1000)
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = self._get_signature(query_string)
        url = f"{config.BINANCE_TESTNET_REST_URL}{endpoint}?{query_string}&signature={signature}"
        
        headers = {
            "X-MBX-APIKEY": self.api_key
        }

        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, timeout=10)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported request method: {method}")

            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error in Binance signed request {method} {endpoint}: {e}")
            raise e

    def get_top_20_usdt_pairs(self):
        """
        Dynamically fetches top 20 USDT pairs by 24h quote volume from Binance Mainnet.
        Falls back to a hardcoded list if API is unreachable.
        """
        try:
            # We use Mainnet for volume rankings since Testnet volume data is sparse
            url = f"{config.BINANCE_MAINNET_REST_URL}/api/v3/ticker/24hr"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            tickers = response.json()
            
            # Filter for USDT pairs, sort by quoteVolume descending
            usdt_tickers = [t for t in tickers if t['symbol'].endswith('USDT') and not t['symbol'].endswith('UPUSDT') and not t['symbol'].endswith('DOWNUSDT')]
            usdt_tickers.sort(key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
            
            limit = config.MAX_SYMBOLS
            top_symbols = [t['symbol'] for t in usdt_tickers[:limit]]
            logger.info(f"Successfully fetched top {len(top_symbols)} USDT trading pairs dynamically.")
            return top_symbols
        except Exception as e:
            logger.warning(f"Failed to fetch top USDT pairs dynamically ({e}). Using hardcoded fallbacks.")
            return self.fallback_pairs[:config.MAX_SYMBOLS]

    def fetch_historical_klines(self, symbol, interval="1m", limit=500, start_time=None):
        """
        Fetches historical klines. Tries Binance Mainnet (no auth needed) first,
        and falls back to Binance Testnet if rate-limited or blocked (e.g. on shared hostings).
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        if start_time:
            params["startTime"] = start_time
            
        # Try Mainnet first
        url_mainnet = f"{config.BINANCE_MAINNET_REST_URL}/api/v3/klines"
        try:
            response = requests.get(url_mainnet, params=params, timeout=10)
            if response.status_code in (418, 429):
                logger.info(f"Mainnet API returned {response.status_code} for {symbol}. Falling back to Testnet...")
                raise requests.exceptions.RequestException(f"Rate limited or teapot: {response.status_code}")
            response.raise_for_status()
            klines = response.json()
        except Exception as e:
            # Fallback to Testnet REST API
            logger.info(f"Error fetching klines from Mainnet for {symbol} ({e}). Trying Testnet...")
            url_testnet = f"{config.BINANCE_TESTNET_REST_URL}/api/v3/klines"
            try:
                response = requests.get(url_testnet, params=params, timeout=10)
                response.raise_for_status()
                klines = response.json()
            except Exception as e_inner:
                logger.error(f"Failed to fetch klines from both Mainnet and Testnet for {symbol}: {e_inner}")
                return []
                
        # Format: [ [open_time, open, high, low, close, volume, close_time, ...], ... ]
        formatted = []
        for k in klines:
            formatted.append({
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5])
            })
        return formatted

    # Signed endpoints
    def get_account_balance(self):
        """Gets account asset balance from Binance Testnet."""
        if self.disable_execution:
            return {"USDT": config.INITIAL_USDT_BALANCE}
            
        try:
            account_info = self._send_signed_request("GET", "/api/v3/account")
            balances = {}
            for asset in account_info.get("balances", []):
                free_val = float(asset.get("free", 0.0))
                locked_val = float(asset.get("locked", 0.0))
                total = free_val + locked_val
                if total > 0:
                    balances[asset["asset"]] = free_val
            return balances
        except Exception as e:
            logger.error(f"Failed to get account balances: {e}")
            return {"USDT": 0.0}

    def place_market_buy(self, symbol, usdt_quantity):
        """Places a market BUY order using quoteOrderQty on Binance Testnet."""
        params = {
            "symbol": symbol,
            "side": "BUY",
            "type": "MARKET",
            "quoteOrderQty": round(usdt_quantity, 4)
        }
        try:
            result = self._send_signed_request("POST", "/api/v3/order", params)
            logger.info(f"Market BUY executed for {symbol}: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to place market BUY order for {symbol}: {e}")
            return None

    def place_market_sell(self, symbol, coin_quantity):
        """Places a market SELL order on Binance Testnet."""
        # Binance has specific lot size filters, so we round carefully.
        # To avoid issues in testnet, we format quantity as string or float depending on asset size.
        params = {
            "symbol": symbol,
            "side": "SELL",
            "type": "MARKET",
            "quantity": float(coin_quantity)
        }
        try:
            result = self._send_signed_request("POST", "/api/v3/order", params)
            logger.info(f"Market SELL executed for {symbol}: {result}")
            return result
        except Exception as e:
            logger.error(f"Failed to place market SELL order for {symbol}: {e}")
            return None

    # WebSocket streams
    def start_websocket_stream(self, symbols, on_candle_update_callback):
        """Starts a background WebSocket connection for multiple symbol streams."""
        # Format streams: e.g. btcusdt@kline_1m/ethusdt@kline_1m/...
        streams = "/".join([f"{s.lower()}@kline_1m" for s in symbols])
        ws_url = f"{config.BINANCE_TESTNET_STREAM_URL}?streams={streams}"
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                stream_name = data.get("stream")
                payload = data.get("data", {})
                
                symbol = payload.get("s")
                kline = payload.get("k", {})
                
                # Check if kline is closed
                is_closed = kline.get("x", False)
                open_time = int(kline.get("t"))
                open_price = float(kline.get("o"))
                high_price = float(kline.get("h"))
                low_price = float(kline.get("l"))
                close_price = float(kline.get("c"))
                volume = float(kline.get("v"))
                
                on_candle_update_callback(symbol, is_closed, open_time, open_price, high_price, low_price, close_price, volume)
            except Exception as e:
                logger.error(f"Error parsing WebSocket message: {e}")

        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            logger.warning(f"WebSocket connection closed: {close_status_code} - {close_msg}")
            self.is_connected = False
            # Reconnect after delay
            threading.Thread(target=self._reconnect, args=(symbols, on_candle_update_callback)).start()

        def on_open(ws):
            logger.info("WebSocket connection established successfully.")
            self.is_connected = True

        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def _reconnect(self, symbols, callback, delay=5):
        """Attempts to reconnect to WebSockets after a network loss."""
        time.sleep(delay)
        logger.info("Attempting WebSocket reconnection...")
        self.start_websocket_stream(symbols, callback)

    def stop_websocket_stream(self):
        if self.ws:
            self.ws.close()
            logger.info("WebSocket connection explicitly stopped.")
