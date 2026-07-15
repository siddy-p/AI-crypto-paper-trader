# AI-Powered Crypto Paper Trading System (Binance Testnet)

An autonomous cryptocurrency paper trading bot built in Python. The system connects to the Binance Testnet API via WebSockets for real-time market data, calculates normalized technical indicators, makes minute-by-minute trading predictions using a daily retrained XGBoost classifier, enforces risk management limits, and presents performance metrics on a glassmorphic dark-mode web dashboard.

---

## Features
- **Live WebSocket Data**: Streams 1-minute candle bars continuously for the top 20 USDT trading pairs.
- **Dynamic Asset Selection**: Dynamically ranks and monitors the top 20 symbols based on 24h trading volume.
- **XGBoost AI Predictor**: Predicts the probability of price increasing by $\ge 0.5\%$ in the next 15 minutes.
- **Scale-Independent Features**: Normalizes indicators (EMA, MACD, Bollinger Bands, ATR) relative to close price to ensure model generalizes across different denominations.
- **Automatic Retraining**: Re-downloads historical data and retrains the model every 24 hours in a background worker thread.
- **Strict Risk Bounds**: Risks at most 1% of total portfolio equity per trade, using a 1% Stop Loss and a 2% Take Profit.
- **Safety Kill Switch**: Instantly bypasses API orders and runs local dry-run simulations when `DISABLE_ORDER_EXECUTION` is `True`.
- **SQLite Database**: Persists transaction logs, candle caches, daily performance metrics, and portfolio value snapshots.
- **Sleek Web Dashboard**: Renders portfolio value history, open positions, win rates, Sharpe ratio, drawdowns, recent trades, and real-time model confidence gauges.

---

## Folder Structure
```
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── README.md
├── .env.example
├── main.py                     # Entry point
├── src/
│   ├── config.py               # Env configurations & endpoints
│   ├── database.py             # SQLite helper functions
│   ├── binance_client.py       # REST and WS client
│   ├── data_manager.py         # Rolling buffers & feature generation
│   ├── indicators.py           # Pandas vectorized formulas
│   ├── model.py                # XGBoost training/inference wrapper
│   └── dashboard.py            # Flask server for API & templates
│       └── templates/
│           └── index.html      # Front-end dashboard UI
└── tests/
    ├── test_indicators.py      # Indicator unit tests
    ├── test_model.py           # ML pipeline tests
    └── test_trader.py          # Trade loop and risk tests
```

---

## Installation & Setup

### Option 1: Native Execution

1. **Clone and Navigate**:
   ```bash
   cd "/Users/siddy/Project AI Crypto Trading"
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**:
   Create a `.env` file from the example template:
   ```bash
   cp .env.example .env
   ```
   Modify `.env` to configure your API keys (get Testnet API keys from [testnet.binance.vision](https://testnet.binance.vision/)). Set `DISABLE_ORDER_EXECUTION=True` for safe simulations or `False` to trade with testnet funds.

5. **Start Bot & Web Dashboard**:
   ```bash
   python main.py
   ```
   Open `http://localhost:5001` in your browser.

---

### Option 2: Docker Compose (Recommended)

1. Ensure Docker Desktop is running.
2. Build and launch the container:
   ```bash
   docker-compose up --build -d
   ```
3. Monitor logs:
   ```bash
   docker-compose logs -f
   ```
4. Access the dashboard at `http://localhost:5001`. Data and SQLite logs will persist in the `trader_data` docker volume.

---

## Running Unit Tests
Execute the test suite using `unittest`:
```bash
python -m unittest discover -s tests
```
The suite verifies indicator vector math, Look-Forward training labels, position sizing, live API overrides, and SL/TP triggers.
