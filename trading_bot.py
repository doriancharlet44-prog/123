"""
═══════════════════════════════════════════════════════════════
    PROFESSIONAL TRADING BOT - MAIN ENGINE
═══════════════════════════════════════════════════════════════
"""

import sys
import time
import logging
import threading
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque
import numpy as np

from database import get_db
from strategies import StrategyManager, Signal
from ml_engine import MachineLearningEngine
from risk_manager import AdvancedRiskManager, RiskLimits

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IBKRConnector:
    """Interactive Brokers connector with graceful fallback to simulation"""

    def __init__(self):
        self.ib = None
        self.connected = False
        self.simulation_mode = True  # Default to simulation

        # Try importing ib-insync
        try:
            from ib_insync import IB, Stock, MarketOrder
            self.IB = IB
            self.Stock = Stock
            self.MarketOrder = MarketOrder
            self.simulation_mode = False
            logger.info("✅ ib-insync available")
        except ImportError:
            logger.warning("⚠️ ib-insync not installed — running in DEMO mode")
            self.simulation_mode = True

        self.host = '127.0.0.1'
        self.port = 7497
        self.client_id = 1

    def connect(self) -> bool:
        if self.simulation_mode:
            logger.info("🎮 DEMO mode active (install TWS + ib-insync for live)")
            self.connected = True
            return True
        try:
            self.ib = self.IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            logger.info(f"✅ Connected to IBKR ({self.host}:{self.port})")
            return True
        except Exception as e:
            logger.error(f"❌ IBKR connection failed: {e}")
            logger.info("🎮 Falling back to DEMO mode")
            self.simulation_mode = True
            self.connected = True
            return True

    def disconnect(self):
        if self.ib and not self.simulation_mode:
            try:
                self.ib.disconnect()
            except Exception:
                pass
        self.connected = False

    def get_price(self, symbol: str) -> Optional[float]:
        if self.simulation_mode:
            base_prices = {
                'AAPL': 185.0, 'MSFT': 415.0, 'GOOGL': 175.0,
                'AMZN': 195.0, 'TSLA': 260.0, 'NVDA': 850.0,
                'META': 520.0, 'JPM': 195.0, 'V': 280.0, 'WMT': 170.0
            }
            base = base_prices.get(symbol, 100.0)
            return round(base * (1 + np.random.randn() * 0.008), 2)
        try:
            contract = self.Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(0.5)
            if ticker.last:
                return float(ticker.last)
            elif ticker.close:
                return float(ticker.close)
            return None
        except Exception as e:
            logger.error(f"❌ Price error for {symbol}: {e}")
            return None

    def place_order(self, symbol: str, quantity: float, action: str) -> bool:
        if self.simulation_mode:
            logger.info(f"🎮 [DEMO] {action} {quantity:.4f} {symbol}")
            return True
        try:
            contract = self.Stock(symbol, 'SMART', 'USD')
            self.ib.qualifyContracts(contract)
            order = self.MarketOrder(action, abs(int(quantity)))
            self.ib.placeOrder(contract, order)
            logger.info(f"✅ Order placed: {action} {quantity:.4f} {symbol}")
            return True
        except Exception as e:
            logger.error(f"❌ Order failed: {e}")
            return False


class TradingBot:
    """Main trading bot engine"""

    def __init__(self):
        self.db = get_db()
        self.load_config()

        self.ibkr = IBKRConnector()
        self.strategy_manager = StrategyManager()
        self.ml_engine = MachineLearningEngine()
        self.risk_manager = AdvancedRiskManager(self.initial_capital)

        self.running = False
        self.paused = False
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                        'NVDA', 'META', 'JPM', 'V', 'WMT']
        self.last_prices = {}
        self.start_time = None
        self.trade_count = 0

        logger.info("═" * 60)
        logger.info("🚀 PROFESSIONAL TRADING BOT INITIALIZED")
        logger.info("═" * 60)

    def load_config(self):
        self.initial_capital = self.db.get_config('initial_capital', 500.0)
        self.trading_mode = self.db.get_config('trading_mode', 'paper')
        self.min_confidence = self.db.get_config('min_confidence', 0.60)
        logger.info(f"💰 Capital: €{self.initial_capital:.2f} | Mode: {self.trading_mode} | Min confidence: {self.min_confidence:.0%}")

    def start(self):
        if self.running:
            return False
        logger.info("🚀 STARTING TRADING BOT")
        if not self.ibkr.connect():
            return False
        self.running = True
        self.start_time = datetime.now()
        threading.Thread(target=self._trading_loop, daemon=True).start()
        threading.Thread(target=self._monitoring_loop, daemon=True).start()
        logger.info("✅ Bot started")
        return True

    def stop(self):
        if not self.running:
            return False
        logger.info("🛑 Stopping bot...")
        self.running = False
        self._close_all_positions()
        self.ibkr.disconnect()
        self._save_metrics()
        logger.info("✅ Bot stopped")
        return True

    def _trading_loop(self):
        logger.info("🔄 Trading loop started")
        while self.running:
            try:
                if self.paused:
                    time.sleep(5)
                    continue
                for symbol in self.symbols:
                    if not self.running:
                        break
                    self._process_symbol(symbol)
                    time.sleep(0.3)
                self._manage_positions()
                time.sleep(5)
            except Exception as e:
                logger.error(f"❌ Trading loop error: {e}", exc_info=True)
                time.sleep(10)

    def _process_symbol(self, symbol: str):
        price = self.ibkr.get_price(symbol)
        if price is None:
            return
        self.strategy_manager.update_all_prices(symbol, price)
        self.last_prices[symbol] = price

        can_trade, reason = self.risk_manager.can_open_new_position()
        if not can_trade:
            return

        market_data = {
            'prices': self.strategy_manager.strategies[0].get_prices(symbol),
            'current_price': price,
            'symbol': symbol
        }
        signal = self.strategy_manager.get_best_signal(symbol, price, market_data)
        if signal is None:
            return

        signal = self.ml_engine.enhance_signal(signal, market_data)
        if signal.confidence < self.min_confidence:
            return

        metrics = self.risk_manager.get_risk_metrics()
        quantity = self.risk_manager.calculate_position_size(
            signal.confidence,
            metrics['win_rate'] / 100,
            metrics['avg_win'],
            metrics['avg_loss'],
            price
        )
        self._open_position(signal, quantity)

    def _open_position(self, signal: Signal, quantity: float):
        if not self.ibkr.place_order(signal.symbol, quantity, signal.direction):
            return
        self.risk_manager.open_position(
            signal.symbol, signal.direction, signal.entry_price,
            quantity, signal.stop_loss, signal.take_profit,
            signal.strategy, signal.confidence
        )
        self.db.insert_trade({
            'symbol': signal.symbol, 'strategy': signal.strategy,
            'direction': signal.direction,
            'entry_time': signal.timestamp.isoformat(),
            'entry_price': signal.entry_price, 'quantity': quantity,
            'stop_loss': signal.stop_loss, 'take_profit': signal.take_profit,
            'confidence': signal.confidence, 'features': signal.features
        })
        self.trade_count += 1
        logger.info(f"✅ OPEN: {signal.direction} {quantity:.4f} {signal.symbol} @ ${signal.entry_price:.2f}")

    def _manage_positions(self):
        for symbol in list(self.risk_manager.open_positions.keys()):
            price = self.ibkr.get_price(symbol)
            if price is None:
                continue
            self.risk_manager.update_position(symbol, price)
            should_close, reason = self.risk_manager.should_close_position(symbol)
            if should_close:
                self._close_position(symbol, price, reason)

    def _close_position(self, symbol: str, exit_price: float, reason: str):
        position = self.risk_manager.open_positions.get(symbol)
        if not position:
            return
        opposite = 'SELL' if position['direction'] == 'BUY' else 'BUY'
        if not self.ibkr.place_order(symbol, position['quantity'], opposite):
            return
        result = self.risk_manager.close_position(symbol, exit_price, reason)
        self.strategy_manager.update_strategy_performance(result['strategy'], result['pnl'] > 0)
        market_data = {
            'prices': self.strategy_manager.strategies[0].get_prices(symbol),
            'current_price': exit_price
        }
        self.ml_engine.learn_from_trade(market_data, result['direction'], result['pnl'])
        logger.info(f"🔚 CLOSE: {symbol} P&L €{result['pnl']:.2f} ({reason})")

    def _close_all_positions(self):
        for symbol in list(self.risk_manager.open_positions.keys()):
            price = self.ibkr.get_price(symbol)
            if price:
                self._close_position(symbol, price, "Bot stopped")

    def _monitoring_loop(self):
        while self.running:
            try:
                time.sleep(60)
                if self.running:
                    self._save_metrics()
            except Exception as e:
                logger.error(f"Monitor error: {e}")

    def _save_metrics(self):
        metrics = self.risk_manager.get_risk_metrics()
        logger.info(f"📊 Capital=€{metrics['current_capital']:.2f} P&L={metrics['total_pnl']:+.2f} WR={metrics['win_rate']:.1f}%")

    def _format_uptime(self) -> str:
        if not self.start_time:
            return "0s"
        delta = datetime.now() - self.start_time
        total_seconds = int(delta.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def get_status(self) -> Dict:
        metrics = self.risk_manager.get_risk_metrics()
        metrics['running'] = self.running
        metrics['paused'] = self.paused
        metrics['connected'] = self.ibkr.connected
        metrics['simulation_mode'] = self.ibkr.simulation_mode
        metrics['trade_count'] = self.trade_count
        metrics['uptime'] = self._format_uptime()

        metrics['open_positions_list'] = [
            {
                'symbol': pos['symbol'],
                'direction': pos['direction'],
                'entry_price': pos['entry_price'],
                'current_price': pos['current_price'],
                'quantity': pos['quantity'],
                'unrealized_pnl': pos['unrealized_pnl'],
                'unrealized_pnl_pct': pos['unrealized_pnl_pct'],
                'strategy': pos['strategy']
            }
            for pos in self.risk_manager.open_positions.values()
        ]
        metrics['strategies'] = self.strategy_manager.get_strategies_status()
        metrics['ml_stats'] = self.ml_engine.get_ml_stats()
        return metrics

    def pause(self):
        self.paused = True
        logger.info("⏸️ Bot paused")

    def resume(self):
        self.paused = False
        logger.info("▶️ Bot resumed")

    def update_config(self, config: Dict):
        for key, value in config.items():
            self.db.set_config(key, value, type(value).__name__)
        self.load_config()
        logger.info("⚙️ Configuration updated")


# Singleton
_bot_instance = None

def get_bot() -> TradingBot:
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = TradingBot()
    return _bot_instance
