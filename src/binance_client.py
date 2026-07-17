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

# ---------------------------------------------------------------------------
# Binance public REST weight limits
#   Hard cap  : 1200 weight / 60-second rolling window
#   Our budget : 1100 weight / 60s  (100 weight safety buffer)
#   klines limit=1000 → 10 weight per call
#   klines limit=500  →  5 weight per call
#   ticker/24hr (all) → 40 weight per call
# ---------------------------------------------------------------------------

class _WeightBucket:
    """
    Sliding-window weight tracker for Binance's IP-level rate limit.

    Before each API call, `consume(weight)` is called:
    - If enough budget remains in the current 60-second window, the call
      proceeds immediately.
    - Otherwise it sleeps for precisely the time until the oldest entry
      falls out of the window, freeing up enough budget — then proceeds.

    This gives the *maximum possible throughput* without ever exceeding
    the limit, replacing all fixed sleeps.
    """

    WINDOW   = 60    # seconds
    BUDGET   = 1100  # weight units per window (hard cap is 1200; 100 safety buffer)

    def __init__(self):
        self._log  = []  # list of (timestamp: float, weight: int)
        self._lock = threading.Lock()

    def consume(self, weight: int):
        """Block until `weight` units can be spent, then record the spend."""
        with self._lock:
            while True:
                now = time.monotonic()
                # Evict entries that have rolled out of the window
                cutoff = now - self.WINDOW
                self._log = [(t, w) for t, w in self._log if t > cutoff]

                used = sum(w for _, w in self._log)
                if used + weight <= self.BUDGET:
                    self._log.append((now, weight))
                    return  # budget available — proceed immediately

                # Not enough budget yet. Sleep until the oldest entry expires.
                oldest_ts = self._log[0][0]
                sleep_for = (oldest_ts + self.WINDOW) - now + 0.05  # 50ms padding
                logger.debug(
                    f"[WeightBucket] Budget {used}/{self.BUDGET} used. "
                    f"Sleeping {sleep_for:.2f}s for window to clear..."
                )
                time.sleep(max(0.05, sleep_for))

    @property
    def used(self) -> int:
        """Current weight consumed in the active window (for logging)."""
        with self._lock:
            now = time.monotonic()
            return sum(w for t, w in self._log if t > now - self.WINDOW)

class BinanceClient:
    def __init__(self):
        self.api_key = config.BINANCE_API_KEY
        self.api_secret = config.BINANCE_API_SECRET
        self.disable_execution = config.DISABLE_ORDER_EXECUTION
        self.ws = None
        self.ws_thread = None
        self.is_connected = False

        # Shared weight bucket — enforces Binance's 1200 weight/min limit
        self._bucket = _WeightBucket()
        
        # Safe fallback list of top 200 USDT pairs by volume (used when Mainnet 24hr ticker is rate-limited)
        self.fallback_pairs = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "SHIBUSDT", "AVAXUSDT", "DOTUSDT",
            "LINKUSDT", "NEARUSDT", "LTCUSDT", "BCHUSDT", "TRXUSDT",
            "ETCUSDT", "XLMUSDT", "UNIUSDT", "FILUSDT", "ATOMUSDT",
            "ICPUSDT", "APTUSDT", "OPUSDT", "ARBUSDT", "INJUSDT",
            "SUIUSDT", "SEIUSDT", "TIAUSDT", "RNDRUSDT", "LDOUSDT",
            "IMXUSDT", "STXUSDT", "RUNEUSDT", "ORDIUSDT", "KASUSDT",
            "WLDUSDT", "PENDLEUSDT", "PYTHUSDT", "JUPUSDT", "WIFUSDT",
            "BONKUSDT", "PEPEUSDT", "FLOKIUSDT", "FETUSDT", "GRTUSDT",
            "AAVEUSDT", "MKRUSDT", "SNXUSDT", "CRVUSDT", "COMPUSDT",
            "GALAUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT", "ENJUSDT",
            "CHZUSDT", "FLOWUSDT", "ALGOUSDT", "XTZUSDT", "EGLDUSDT",
            "HBARUSDT", "ONEUSDT", "XMRUSDT", "ZECUSDT", "DASHUSDT",
            "BATUSDT", "ZRXUSDT", "YFIUSDT", "SUSHIUSDT", "1INCHUSDT",
            "CAKEUSDT", "ANKRUSDT", "CELRUSDT", "IOSTUSDT", "HOTUSDT",
            "VETUSDT", "XVGUSDT", "ONTUSDT", "ZILUSDT", "IOTAUSDT",
            "NEOUSDT", "WAVESUSDT", "QTUMUSDT", "ICXUSDT", "RVNUSDT",
            "SCUSDT", "STORJUSDT", "BANDUSDT", "KNCUSDT", "BALUSDT",
            "OCEANUSDT", "CKBUSDT", "KAVAUSDT", "CTSIUSDT", "NKNUSDT",
            "STMXUSDT", "COTIUSDT", "BLZUSDT", "DUSKUSDT", "MTLUSDT",
            "DOCKUSDT", "PHBUSDT", "BEAMUSDT", "HOOKUSDT", "MAGICUSDT",
            "HIGHUSDT", "PERPUSDT", "DYDXUSDT", "GMXUSDT", "UMAUSDT",
            "TRUUSDT", "LRCUSDT", "YFIIUSDT", "RLCUSDT", "REPUSDT",
            "CVCUSDT", "QNTUSDT", "IDUSDT", "ARKUSDT", "AGLDUSDT",
            "ENSUSDT", "DARUSDT", "POWRUSDT", "REQUSDT", "FUNUSDT",
            "GLMRUSDT", "SPELLUSDT", "WOOUSDT", "FLMUSDT", "SFPUSDT",
            "DGBUSDT", "BAKEUSDT", "TVKUSDT", "XECUSDT", "ELFUSDT",
            "MBOXUSDT", "CFXUSDT", "ALPHAUSDT", "TORNUSDT", "CTKUSDT",
            "RADUSDT", "HARDUSDT", "ORNUSDT", "MDTUSDT", "LITUSDT",
            "PAXGUSDT", "RAYUSDT", "SRMUSDT", "PORTOUSDT", "JUVUSDT",
            "PSGUSDT", "ACHUSDT", "WINGUSDT", "SUPERUSDT", "BTTCUSDT",
            "XNOUSDT", "WRXUSDT", "TWTUSDT", "TLMUSDT", "ASTRUSDT",
            "ROSEUSDT", "NUUSDT", "LPTUSDT", "XVSUSDT", "ATMUSDT",
            "ASRUSDT", "ACMUSDT", "PHAUSDT", "AUDAUSDT", "BNTUSDT",
            "MTAUSDT", "ADXUSDT", "MASKUSDT", "SXPUSDT", "KLAYUSDT",
            "CELOUSDT", "AXLUSDT", "ALTUSDT", "JUPUSDT", "DYMUSDT",
            "MNTUSDT", "BLURUSDT", "AIDOGEUSDT", "AGIXUSDT", "CTXCUSDT",
            "MAVUSDT", "PENDLEUSDT", "ARKMUSDT", "CYBERUSDT", "UMAUSDT",
        ]

    def _safe_get(self, url, params=None, weight=1, max_retries=3):
        """
        Rate-limit-aware GET wrapper for Binance public endpoints.

        Before making the request, consumes `weight` units from the sliding-window
        weight bucket so we never exceed Binance's 1200 weight/min limit.

        - On HTTP 429 (rate limited): reads the Retry-After header and waits.
        - On HTTP 418 (IP banned):    waits the Retry-After duration + 60s buffer,
          then raises so the caller can decide whether to abort or fallback.
        - All other transient errors are retried up to max_retries times with
          exponential backoff (2s → 4s → 8s).

        Returns the requests.Response object on success, or raises on failure.
        """
        for attempt in range(1, max_retries + 1):
            # Block here until we have budget — this is the main throttle mechanism.
            self._bucket.consume(weight)

            try:
                response = requests.get(url, params=params, timeout=10)

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 30))
                    wait = retry_after + 5
                    logger.warning(
                        f"Binance 429 rate limit (attempt {attempt}/{max_retries}). "
                        f"Waiting {wait}s (Retry-After={retry_after}s)..."
                    )
                    time.sleep(wait)
                    continue  # retry — bucket.consume will fire again next loop

                if response.status_code == 418:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    wait = retry_after + 60
                    logger.error(
                        f"Binance 418 IP ban! Waiting {wait}s for ban to expire "
                        f"(Retry-After={retry_after}s + 60s buffer), then retrying..."
                    )
                    time.sleep(wait)
                    continue  # retry after ban expires — do NOT fall back to Testnet

                response.raise_for_status()
                return response

            except requests.exceptions.HTTPError:
                raise  # propagate HTTP errors immediately
            except Exception as e:
                backoff = 2 ** attempt  # 2s, 4s, 8s
                logger.warning(
                    f"Request to {url} failed (attempt {attempt}/{max_retries}): {e}. "
                    f"Retrying in {backoff}s..."
                )
                if attempt == max_retries:
                    raise
                time.sleep(backoff)

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
            # ticker/24hr with no symbol filter costs 40 weight
            response = self._safe_get(url, weight=40)
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
            
        # Weight cost: limit=1000 → 10, limit=500 → 5, anything else → ceil(limit/100)
        kline_weight = max(1, (limit + 99) // 100)

        # Try Mainnet first (weight bucket ensures we never exceed 1100 weight/min)
        url_mainnet = f"{config.BINANCE_MAINNET_REST_URL}/api/v3/klines"
        try:
            response = self._safe_get(url_mainnet, params=params, weight=kline_weight)
            klines = response.json()
        except Exception as e:
            # Fallback to Testnet REST API
            logger.info(f"Error fetching klines from Mainnet for {symbol} ({e}). Trying Testnet...")
            url_testnet = f"{config.BINANCE_TESTNET_REST_URL}/api/v3/klines"
            try:
                response = self._safe_get(url_testnet, params=params, weight=kline_weight)
                klines = response.json()
            except Exception as e_inner:
                # 400 Bad Request means the token pair is not listed/supported on the Testnet API
                if isinstance(e_inner, requests.exceptions.HTTPError) and e_inner.response is not None and e_inner.response.status_code == 400:
                    logger.info(f"Symbol {symbol} is not supported on Binance Testnet (400 Bad Request).")
                else:
                    logger.warning(f"Failed to fetch klines from both Mainnet and Testnet for {symbol}: {e_inner}")
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
        """Starts a background WebSocket on Binance MAINNET for real volume/tick data."""
        # Mainnet stream URL — real market data with actual volume
        streams = "/".join([f"{s.lower()}@kline_1m" for s in symbols])
        ws_url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        
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

    def _reconnect(self, symbols, callback, attempt=1):
        """Exponential backoff reconnect: 5s, 10s, 20s, 40s, capped at 60s."""
        delay = min(5 * (2 ** (attempt - 1)), 60)
        logger.warning(f"WebSocket reconnect attempt #{attempt} in {delay}s...")
        time.sleep(delay)
        self.start_websocket_stream(symbols, callback)

    def stop_websocket_stream(self):
        if self.ws:
            self.ws.close()
            logger.info("WebSocket connection explicitly stopped.")
