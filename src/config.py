import os
from pathlib import Path
from dotenv import load_dotenv

# Load env variables from .env file if it exists
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Database config
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "trading_system.db"))

# Binance Testnet API credentials
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# Safety configuration flag: Set to True to disable all order execution instantly
# (Trades will be simulated in memory and SQLite, but no requests will be sent to the Binance API to buy/sell).
DISABLE_ORDER_EXECUTION = os.getenv("DISABLE_ORDER_EXECUTION", "True").lower() in ("true", "1", "yes")

# Binance API endpoints
BINANCE_TESTNET_REST_URL = "https://testnet.binance.vision"
BINANCE_TESTNET_WS_URL = "wss://stream.testnet.binance.vision/ws"
BINANCE_TESTNET_STREAM_URL = "wss://stream.testnet.binance.vision/stream"

# Binance Mainnet API (used strictly for fetching public historical candles for training)
BINANCE_MAINNET_REST_URL = "https://api.binance.com"

# Model Config
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "xgboost_model.json"))

# Web Dashboard Config
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5001"))
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")

# Initial balance for local simulation if Binance Testnet is disabled or credentials missing
INITIAL_USDT_BALANCE = float(os.getenv("INITIAL_USDT_BALANCE", "1000.0"))

# PostgreSQL Database URL
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Maximum number of symbols to fetch and track dynamically
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "200"))

# Trading risk parameters (all tunable via Render environment variables)
POSITION_SIZE_USDT   = float(os.getenv("POSITION_SIZE_USDT",   "100.0"))  # Fixed USDT per trade
MAX_OPEN_POSITIONS   = int(os.getenv("MAX_OPEN_POSITIONS",   "5"))        # Max simultaneous open trades ($500 max exposure)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "60.0"))   # Min model confidence to enter a trade (%)
STOP_LOSS_PCT        = float(os.getenv("STOP_LOSS_PCT",        "0.015"))   # 1.5% SL
TAKE_PROFIT_PCT      = float(os.getenv("TAKE_PROFIT_PCT",      "0.030"))   # 3.0% TP (2:1 R:R ratio)
