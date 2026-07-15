import logging
import sys
import time
import threading
from src.binance_client import BinanceClient
from src.model import CryptoModelManager
from src.trader import Trader
from src.dashboard import create_app, run_dashboard

# Configure rich console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("main")

def main():
    logger.info("Initializing Antigravity AI Crypto Trading System...")
    
    # 1. Initialize core system components
    binance_client = BinanceClient()
    model_manager = CryptoModelManager()
    trader = Trader(binance_client, model_manager)
    
    # 2. Initialize Flask Dashboard App immediately
    flask_app = create_app(trader)
    
    # 3. Spin up dashboard in a separate concurrent thread immediately
    # This binds the port (5001) in <1s, passing Render's deployment health checks.
    logger.info("Starting dashboard web server thread...")
    dashboard_thread = threading.Thread(
        target=run_dashboard,
        args=(flask_app,),
        daemon=True
    )
    dashboard_thread.start()
    
    # Give the web server a moment to bind and launch
    time.sleep(1.0)
    
    # 4. Bootstrap data cache, portfolio value, and machine learning models in background
    try:
        trader.bootstrap_system()
    except Exception as e:
        logger.critical(f"Critical failure during system bootstrap: {e}")
        sys.exit(1)
        
    # 5. Define callback to handle real-time WebSocket ticks
    def handle_websocket_candle_update(symbol, is_closed, open_time, open_price, high_price, low_price, close_price, volume):
        trader.handle_live_tick(
            symbol=symbol,
            is_closed=is_closed,
            open_time=open_time,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            close_price=close_price,
            volume=volume
        )
        
    # 6. Start WebSocket stream connection in the background
    logger.info("Starting live market data WebSocket stream...")
    binance_client.start_websocket_stream(trader.symbols, handle_websocket_candle_update)
    trader.is_running = True
    
    logger.info("Antigravity trading system is fully operational.")
    
    # 7. Main loop to keep parent thread alive and handle interrupts
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("System shutdown request received. Terminating bot...")
        trader.is_running = False
        binance_client.stop_websocket_stream()
        logger.info("System successfully shut down.")

if __name__ == "__main__":
    main()
