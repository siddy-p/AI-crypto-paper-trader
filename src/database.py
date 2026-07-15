import sqlite3
import datetime
import math
from src.config import DB_PATH

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Create trades table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,          -- BUY
        entry_price REAL NOT NULL,
        exit_price REAL,
        entry_time TEXT NOT NULL,
        exit_time TEXT,
        quantity REAL NOT NULL,
        confidence REAL NOT NULL,
        pnl REAL DEFAULT 0.0,
        status TEXT NOT NULL,        -- OPEN, CLOSED
        stop_loss REAL NOT NULL,
        take_profit REAL NOT NULL,
        exit_reason TEXT             -- STOP_LOSS, TAKE_PROFIT, MANUAL, SAFETY_DISABLE, RETRAIN
    )
    """)
    
    # 2. Create portfolio history table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolio_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        total_value REAL NOT NULL,
        available_usdt REAL NOT NULL
    )
    """)
    
    # 3. Create candle cache table
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
    
    # 4. Create daily performance reports table
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
    cursor = conn.cursor()
    cursor.executemany("""
    INSERT OR REPLACE INTO candles (symbol, open_time, open, high, low, close, volume)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, candles_data)
    conn.commit()
    conn.close()

def get_latest_candle_time(symbol):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(open_time) FROM candles WHERE symbol = ?", (symbol,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row and row[0] is not None else 0

def get_cached_candles(symbol, limit=10000):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT open_time, open, high, low, close, volume 
    FROM candles 
    WHERE symbol = ? 
    ORDER BY open_time ASC
    """, (symbol,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Trade logging helpers
def add_trade(symbol, side, entry_price, quantity, confidence, stop_loss, take_profit):
    conn = get_db_connection()
    cursor = conn.cursor()
    entry_time = datetime.datetime.now().isoformat()
    cursor.execute("""
    INSERT INTO trades (symbol, side, entry_price, entry_time, quantity, confidence, stop_loss, take_profit, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
    """, (symbol, side, entry_price, entry_time, quantity, confidence, stop_loss, take_profit))
    trade_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return trade_id

def close_trade(trade_id, exit_price, pnl, exit_reason):
    conn = get_db_connection()
    cursor = conn.cursor()
    exit_time = datetime.datetime.now().isoformat()
    cursor.execute("""
    UPDATE trades 
    SET exit_price = ?, exit_time = ?, pnl = ?, status = 'CLOSED', exit_reason = ?
    WHERE id = ?
    """, (exit_price, exit_time, pnl, exit_reason, trade_id))
    conn.commit()
    conn.close()

def get_open_trades():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trades WHERE status = 'OPEN'")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_trade_history(limit=100):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Portfolio logging helpers
def log_portfolio(total_value, available_usdt):
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    cursor.execute("""
    INSERT INTO portfolio_history (timestamp, total_value, available_usdt)
    VALUES (?, ?, ?)
    """, (timestamp, total_value, available_usdt))
    conn.commit()
    conn.close()

def get_portfolio_history(limit=1000):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM portfolio_history ORDER BY timestamp ASC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

# Statistics and metrics calculations
def calculate_metrics():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Fetch all closed trades to compute statistics
    cursor.execute("SELECT * FROM trades WHERE status = 'CLOSED'")
    closed_trades = [dict(row) for row in cursor.fetchall()]
    
    # Fetch latest portfolio values
    cursor.execute("SELECT total_value, available_usdt FROM portfolio_history ORDER BY id DESC LIMIT 1")
    latest_portfolio = cursor.fetchone()
    conn.close()
    
    portfolio_value = latest_portfolio['total_value'] if latest_portfolio else 10000.0
    available_usdt = latest_portfolio['available_usdt'] if latest_portfolio else 10000.0
    
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
    # We compute periodic returns based on logged portfolio values
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
                # Annualize the Sharpe ratio assuming portfolio snapshots are recorded frequently.
                # Since we log every minute, annualized factor is sqrt(60 * 24 * 365) = sqrt(525600)
                # To be conservative and avoid massive swings from minute data, we can use the simple ratio
                # or multiply by sqrt(252 * 288) depending on interval. Let's do the simple Sharpe of returns first,
                # then scale it safely. A common way for high-frequency is: Sharpe = (mean / std) * sqrt(periods_per_year).
                # Let's assume hourly log updates or just standard 1-min intervals. Let's multiply by sqrt(525600) to annualize,
                # but capped to avoid extreme ratios.
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
    """
    Computes performance metrics for a specific date and logs it to daily_reports.
    date_str: YYYY-MM-DD
    """
    if date_str is None:
        date_str = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        
    metrics = calculate_metrics()
    
    conn = get_db_connection()
    cursor = conn.cursor()
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
    conn.close()

def get_daily_reports(limit=30):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM daily_reports ORDER BY date DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]
