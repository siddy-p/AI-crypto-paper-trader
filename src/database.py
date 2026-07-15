import sqlite3
import psycopg2
import psycopg2.extras
import datetime
import math
import logging
from src import config

logger = logging.getLogger(__name__)

IS_POSTGRES = bool(config.DATABASE_URL)

def get_db_connection():
    if IS_POSTGRES:
        conn = psycopg2.connect(config.DATABASE_URL)
        return conn
    else:
        conn = sqlite3.connect(config.DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

def execute_query(query, params=None):
    """Executes a SELECT query and returns rows as a list of dicts."""
    if params is None:
        params = ()
    conn = get_db_connection()
    try:
        if IS_POSTGRES:
            # Convert SQLite placeholders (?) to Postgres placeholders (%s)
            query = query.replace("?", "%s")
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        else:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    finally:
        conn.close()

def execute_write(query, params=None, return_last_id=False):
    """Executes an INSERT/UPDATE/DELETE query and commits."""
    if params is None:
        params = ()
    conn = get_db_connection()
    try:
        if IS_POSTGRES:
            query = query.replace("?", "%s")
            if return_last_id and "INSERT" in query.upper() and "RETURNING" not in query.upper():
                query += " RETURNING id"
            cursor = conn.cursor()
            cursor.execute(query, params)
            last_id = None
            if return_last_id and "RETURNING" in query.upper():
                last_id = cursor.fetchone()[0]
            conn.commit()
            return last_id
        else:
            cursor = conn.cursor()
            cursor.execute(query, params)
            last_id = cursor.lastrowid if return_last_id else None
            conn.commit()
            return last_id
    finally:
        conn.close()

def init_db():
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if IS_POSTGRES:
            # Postgres Schema
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                side VARCHAR(10) NOT NULL,
                entry_price DOUBLE PRECISION NOT NULL,
                exit_price DOUBLE PRECISION,
                entry_time VARCHAR(50) NOT NULL,
                exit_time VARCHAR(50),
                quantity DOUBLE PRECISION NOT NULL,
                confidence DOUBLE PRECISION NOT NULL,
                pnl DOUBLE PRECISION DEFAULT 0.0,
                status VARCHAR(10) NOT NULL,
                stop_loss DOUBLE PRECISION NOT NULL,
                take_profit DOUBLE PRECISION NOT NULL,
                exit_reason VARCHAR(50)
            )
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id SERIAL PRIMARY KEY,
                timestamp VARCHAR(50) NOT NULL,
                total_value DOUBLE PRECISION NOT NULL,
                available_usdt DOUBLE PRECISION NOT NULL
            )
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol VARCHAR(20) NOT NULL,
                open_time BIGINT NOT NULL,
                open DOUBLE PRECISION NOT NULL,
                high DOUBLE PRECISION NOT NULL,
                low DOUBLE PRECISION NOT NULL,
                close DOUBLE PRECISION NOT NULL,
                volume DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (symbol, open_time)
            )
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_reports (
                date VARCHAR(20) PRIMARY KEY,
                total_trades INTEGER NOT NULL,
                win_rate DOUBLE PRECISION NOT NULL,
                total_pnl DOUBLE PRECISION NOT NULL,
                portfolio_value DOUBLE PRECISION NOT NULL,
                max_drawdown DOUBLE PRECISION NOT NULL,
                sharpe_ratio DOUBLE PRECISION NOT NULL
            )
            """)
        else:
            # SQLite Schema
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                quantity REAL NOT NULL,
                confidence REAL NOT NULL,
                pnl REAL DEFAULT 0.0,
                status TEXT NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                exit_reason TEXT
            )
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                total_value REAL NOT NULL,
                available_usdt REAL NOT NULL
            )
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL,
                open_time INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (symbol, open_time)
            )
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_reports (
                date TEXT PRIMARY KEY,
                total_trades INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                total_pnl REAL NOT NULL,
                portfolio_value REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                sharpe_ratio REAL NOT NULL
            )
            """)
        conn.commit()
    finally:
        conn.close()

# Candle cache helpers
def save_candles(candles_data):
    """
    Saves a list of candles to the database.
    candles_data: list of tuples (symbol, open_time, open, high, low, close, volume)
    """
    if not candles_data:
        return
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if IS_POSTGRES:
            cursor.executemany("""
            INSERT INTO candles (symbol, open_time, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (symbol, open_time) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume
            """, candles_data)
        else:
            cursor.executemany("""
            INSERT OR REPLACE INTO candles (symbol, open_time, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, candles_data)
        conn.commit()
    finally:
        conn.close()

def get_latest_candle_time(symbol):
    rows = execute_query("SELECT MAX(open_time) AS max_val FROM candles WHERE symbol = ?", (symbol,))
    if rows and rows[0]["max_val"] is not None:
        return int(rows[0]["max_val"])
    return 0

def get_cached_candles(symbol, limit=10000):
    return execute_query("""
    SELECT open_time, open, high, low, close, volume 
    FROM candles 
    WHERE symbol = ? 
    ORDER BY open_time ASC
    """, (symbol,))

# Trade logging helpers
def add_trade(symbol, side, entry_price, quantity, confidence, stop_loss, take_profit):
    entry_time = datetime.datetime.now().isoformat()
    trade_id = execute_write("""
    INSERT INTO trades (symbol, side, entry_price, entry_time, quantity, confidence, stop_loss, take_profit, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
    """, (symbol, side, entry_price, entry_time, quantity, confidence, stop_loss, take_profit), return_last_id=True)
    return trade_id

def close_trade(trade_id, exit_price, pnl, exit_reason):
    exit_time = datetime.datetime.now().isoformat()
    execute_write("""
    UPDATE trades 
    SET exit_price = ?, exit_time = ?, pnl = ?, status = 'CLOSED', exit_reason = ?
    WHERE id = ?
    """, (exit_price, exit_time, pnl, exit_reason, trade_id))

def get_open_trades():
    return execute_query("SELECT * FROM trades WHERE status = 'OPEN'")

def get_trade_history(limit=100):
    return execute_query("SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?", (limit,))

# Portfolio logging helpers
def log_portfolio(total_value, available_usdt):
    timestamp = datetime.datetime.now().isoformat()
    execute_write("""
    INSERT INTO portfolio_history (timestamp, total_value, available_usdt)
    VALUES (?, ?, ?)
    """, (timestamp, total_value, available_usdt))

def get_portfolio_history(limit=1000):
    return execute_query("SELECT * FROM portfolio_history ORDER BY timestamp ASC LIMIT ?", (limit,))

# Statistics and metrics calculations
def calculate_metrics():
    closed_trades = execute_query("SELECT * FROM trades WHERE status = 'CLOSED'")
    latest_portfolio = execute_query("SELECT total_value, available_usdt FROM portfolio_history ORDER BY id DESC LIMIT 1")
    
    portfolio_value = latest_portfolio[0]['total_value'] if latest_portfolio else config.INITIAL_USDT_BALANCE
    available_usdt = latest_portfolio[0]['available_usdt'] if latest_portfolio else config.INITIAL_USDT_BALANCE
    
    total_trades = len(closed_trades)
    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "portfolio_value": portfolio_value,
            "available_usdt": available_usdt,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0
        }
        
    wins = sum(1 for t in closed_trades if t['pnl'] > 0)
    win_rate = (wins / total_trades) * 100.0
    total_pnl = sum(t['pnl'] for t in closed_trades)
    
    # Max Drawdown Calculation
    history = get_portfolio_history()
    max_drawdown = 0.0
    if history:
        peak = history[0]['total_value']
        for h in history:
            val = h['total_value']
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100.0 if peak > 0 else 0.0
            if dd > max_drawdown:
                max_drawdown = dd
                
    # Sharpe Ratio Calculation
    sharpe_ratio = 0.0
    if len(history) > 5:
        returns = []
        for i in range(1, len(history)):
            prev = history[i-1]['total_value']
            curr = history[i]['total_value']
            if prev > 0:
                returns.append((curr - prev) / prev)
        
        if returns:
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            std_dev = math.sqrt(variance)
            
            if std_dev > 0:
                sharpe_ratio = (avg_return / std_dev) * math.sqrt(525600)
            else:
                sharpe_ratio = 0.0
                
    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 4),
        "portfolio_value": round(portfolio_value, 2),
        "available_usdt": round(available_usdt, 2),
        "max_drawdown": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe_ratio, 2)
    }

# Daily report functions
def generate_daily_report(date_str=None):
    if date_str is None:
        date_str = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        
    metrics = calculate_metrics()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if IS_POSTGRES:
            cursor.execute("""
            INSERT INTO daily_reports (date, total_trades, win_rate, total_pnl, portfolio_value, max_drawdown, sharpe_ratio)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET
                total_trades = EXCLUDED.total_trades,
                win_rate = EXCLUDED.win_rate,
                total_pnl = EXCLUDED.total_pnl,
                portfolio_value = EXCLUDED.portfolio_value,
                max_drawdown = EXCLUDED.max_drawdown,
                sharpe_ratio = EXCLUDED.sharpe_ratio
            """, (
                date_str, 
                metrics['total_trades'], 
                metrics['win_rate'], 
                metrics['total_pnl'], 
                metrics['portfolio_value'], 
                metrics['max_drawdown'], 
                metrics['sharpe_ratio']
            ))
        else:
            cursor.execute("""
            INSERT OR REPLACE INTO daily_reports (date, total_trades, win_rate, total_pnl, portfolio_value, max_drawdown, sharpe_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                date_str, 
                metrics['total_trades'], 
                metrics['win_rate'], 
                metrics['total_pnl'], 
                metrics['portfolio_value'], 
                metrics['max_drawdown'], 
                metrics['sharpe_ratio']
            ))
        conn.commit()
    finally:
        conn.close()

def get_daily_reports(limit=30):
    return execute_query("SELECT * FROM daily_reports ORDER BY date DESC LIMIT ?", (limit,))
