import logging
import datetime
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
                unrealized_pnl = (current_price - pos["entry_price"]) * pos["quantity"]
                pnl_pct = ((current_price - pos["entry_price"]) / pos["entry_price"]) * 100.0
                
                enriched_positions.append({
                    "id": pos["id"],
                    "symbol": symbol,
                    "side": pos["side"],
                    "entry_price": pos["entry_price"],
                    "current_price": current_price,
                    "quantity": pos["quantity"],
                    "confidence": pos["confidence"],
                    "stop_loss": pos["stop_loss"],
                    "take_profit": pos["take_profit"],
                    "pnl": round(unrealized_pnl, 4),
                    "pnl_pct": round(pnl_pct, 2)
                })
            
            return jsonify({
                "success": True,
                "metrics": metrics,
                "open_positions": enriched_positions,
                "recent_trades": recent_trades,
                "model_predictions": model_predictions,
                "safety_disabled": config.DISABLE_ORDER_EXECUTION,
                "last_retrain_date": str(trader_bot.last_retrain_date) if trader_bot.last_retrain_date else "Never",
                "system_status": "RUNNING" if trader_bot.is_running else "STOPPED"
            })
        except Exception as e:
            logger.error(f"Error serving status API: {e}")
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

    return app

def run_dashboard(app, host=config.DASHBOARD_HOST, port=config.DASHBOARD_PORT):
    logger.info(f"Starting web dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)
