"""
═══════════════════════════════════════════════════════════════
    ADVANCED RISK MANAGEMENT - INSTITUTIONAL GRADE
═══════════════════════════════════════════════════════════════

Gestion des risques niveau institutionnel
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Limites de risque"""
    max_position_size_pct: float = 10.0  # 10% max per position
    max_portfolio_heat: float = 20.0  # 20% total exposure
    max_drawdown_pct: float = 15.0  # 15% max drawdown
    max_daily_loss_pct: float = 5.0  # 5% max daily loss
    max_correlation: float = 0.7  # Max correlation between positions
    min_win_rate: float = 0.45  # Minimum win rate to keep trading


class AdvancedRiskManager:
    """Gestionnaire de risque avancé"""
    
    def __init__(self, initial_capital: float, limits: Optional[RiskLimits] = None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        self.limits = limits or RiskLimits()
        
        # Risk tracking
        self.daily_pnl = {}
        self.equity_curve = deque(maxlen=252)  # 1 year
        self.equity_curve.append(initial_capital)
        
        # Position tracking
        self.open_positions = {}  # {symbol: position_info}
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
        # Risk metrics history
        self.risk_metrics_history = deque(maxlen=30)
        
        logger.info("🛡️ Advanced Risk Manager initialized")
    
    def calculate_position_size(self, 
                               signal_confidence: float,
                               win_rate: float,
                               avg_win: float,
                               avg_loss: float,
                               current_price: float) -> float:
        """Calcule la taille de position optimale (Kelly Criterion dynamique)"""
        
        # Kelly Criterion: f = (p*W - q*L) / W
        # where p = win probability, q = loss probability, W = avg win, L = avg loss
        
        if avg_loss <= 0 or avg_win <= 0:
            # Fallback to fixed percentage
            kelly_fraction = 0.05
        else:
            p = win_rate
            q = 1 - win_rate
            kelly_fraction = (p * avg_win - q * avg_loss) / avg_win
        
        # Apply confidence adjustment
        kelly_fraction *= signal_confidence
        
        # Half-Kelly for safety
        kelly_fraction *= 0.5
        
        # Apply limits
        kelly_fraction = max(0.01, min(kelly_fraction, self.limits.max_position_size_pct / 100))
        
        # Calculate position value
        position_value = self.current_capital * kelly_fraction
        
        # Convert to quantity
        quantity = position_value / current_price
        
        logger.info(f"💰 Kelly sizing: {kelly_fraction*100:.2f}% = ${position_value:.2f} ({quantity:.4f} shares)")
        
        return quantity
    
    def check_portfolio_heat(self) -> Tuple[bool, float]:
        """Vérifie la chaleur du portfolio (exposition totale)"""
        total_exposure = 0.0
        
        for symbol, position in self.open_positions.items():
            exposure = abs(position['quantity'] * position['current_price'])
            total_exposure += exposure
        
        heat_pct = (total_exposure / self.current_capital) * 100
        
        can_trade = heat_pct < self.limits.max_portfolio_heat
        
        if not can_trade:
            logger.warning(f"🔥 Portfolio heat too high: {heat_pct:.1f}% (limit: {self.limits.max_portfolio_heat:.1f}%)")
        
        return can_trade, heat_pct
    
    def check_drawdown(self) -> Tuple[bool, float]:
        """Vérifie le drawdown actuel"""
        if self.peak_capital == 0:
            return True, 0.0
        
        current_drawdown = ((self.peak_capital - self.current_capital) / self.peak_capital) * 100
        
        can_trade = current_drawdown < self.limits.max_drawdown_pct
        
        if not can_trade:
            logger.error(f"📉 Drawdown limit reached: {current_drawdown:.1f}% (limit: {self.limits.max_drawdown_pct:.1f}%)")
        
        return can_trade, current_drawdown
    
    def check_daily_loss(self) -> Tuple[bool, float]:
        """Vérifie les pertes journalières"""
        today = datetime.now().date().isoformat()
        
        if today not in self.daily_pnl:
            return True, 0.0
        
        daily_loss_pct = (self.daily_pnl[today] / self.initial_capital) * 100
        
        can_trade = daily_loss_pct > -self.limits.max_daily_loss_pct
        
        if not can_trade:
            logger.error(f"💸 Daily loss limit reached: {daily_loss_pct:.1f}% (limit: -{self.limits.max_daily_loss_pct:.1f}%)")
        
        return can_trade, daily_loss_pct
    
    def check_win_rate(self) -> Tuple[bool, float]:
        """Vérifie le win rate"""
        if self.total_trades < 20:  # Need minimum trades
            return True, 0.5
        
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        can_trade = win_rate >= self.limits.min_win_rate
        
        if not can_trade:
            logger.error(f"📊 Win rate too low: {win_rate*100:.1f}% (min: {self.limits.min_win_rate*100:.1f}%)")
        
        return can_trade, win_rate
    
    def can_open_new_position(self) -> Tuple[bool, str]:
        """Vérifie si on peut ouvrir une nouvelle position"""
        # Check all risk limits
        checks = [
            self.check_portfolio_heat(),
            self.check_drawdown(),
            self.check_daily_loss(),
            self.check_win_rate()
        ]
        
        for can_trade, value in checks:
            if not can_trade:
                return False, f"Risk limit exceeded"
        
        return True, "All checks passed"
    
    def calculate_stop_loss(self, entry_price: float, direction: str, 
                           atr: float, risk_pct: float = 0.02) -> float:
        """Calcule le stop loss optimal"""
        # Use ATR-based stop loss
        if direction == 'BUY':
            # Stop below entry
            stop_loss = entry_price - (2 * atr)
            
            # Ensure it respects risk percentage
            max_loss = entry_price * risk_pct
            stop_loss = max(stop_loss, entry_price - max_loss)
        else:
            # Stop above entry
            stop_loss = entry_price + (2 * atr)
            
            max_loss = entry_price * risk_pct
            stop_loss = min(stop_loss, entry_price + max_loss)
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, 
                             direction: str, risk_reward_ratio: float = 2.5) -> float:
        """Calcule le take profit optimal"""
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio
        
        if direction == 'BUY':
            take_profit = entry_price + reward
        else:
            take_profit = entry_price - reward
        
        return take_profit
    
    def update_position(self, symbol: str, current_price: float):
        """Met à jour une position existante"""
        if symbol not in self.open_positions:
            return
        
        position = self.open_positions[symbol]
        position['current_price'] = current_price
        
        # Calculate P&L
        if position['direction'] == 'BUY':
            pnl = (current_price - position['entry_price']) * position['quantity']
            pnl_pct = (current_price / position['entry_price'] - 1) * 100
        else:
            pnl = (position['entry_price'] - current_price) * position['quantity']
            pnl_pct = (1 - current_price / position['entry_price']) * 100
        
        position['unrealized_pnl'] = pnl
        position['unrealized_pnl_pct'] = pnl_pct
        
        # Check trailing stop
        if 'trailing_stop' in position:
            self._update_trailing_stop(symbol, current_price)
        
        # Check profit locks
        self._check_profit_locks(symbol, current_price)
    
    def _update_trailing_stop(self, symbol: str, current_price: float):
        """Met à jour le trailing stop"""
        position = self.open_positions[symbol]
        
        if position['direction'] == 'BUY':
            # Trail upward
            new_stop = current_price * (1 - position['trailing_pct'])
            if new_stop > position['stop_loss']:
                position['stop_loss'] = new_stop
                logger.info(f"📈 Trailing stop updated for {symbol}: ${new_stop:.2f}")
        else:
            # Trail downward
            new_stop = current_price * (1 + position['trailing_pct'])
            if new_stop < position['stop_loss']:
                position['stop_loss'] = new_stop
                logger.info(f"📉 Trailing stop updated for {symbol}: ${new_stop:.2f}")
    
    def _check_profit_locks(self, symbol: str, current_price: float):
        """Vérifie et applique les profit locks progressifs"""
        position = self.open_positions[symbol]
        
        profit_pct = position.get('unrealized_pnl_pct', 0)
        
        # Progressive profit locks
        profit_locks = [
            (3.0, 1.0),   # At +3%, lock +1%
            (5.0, 2.0),   # At +5%, lock +2%
            (10.0, 5.0)   # At +10%, lock +5%
        ]
        
        for trigger_pct, lock_pct in profit_locks:
            if profit_pct >= trigger_pct:
                lock_price = position['entry_price'] * (1 + lock_pct / 100)
                
                if position['direction'] == 'BUY':
                    if lock_price > position['stop_loss']:
                        position['stop_loss'] = lock_price
                        logger.info(f"🔒 Profit lock activated for {symbol}: {lock_pct:.1f}% @ ${lock_price:.2f}")
                else:
                    if lock_price < position['stop_loss']:
                        position['stop_loss'] = lock_price
                        logger.info(f"🔒 Profit lock activated for {symbol}: {lock_pct:.1f}% @ ${lock_price:.2f}")
    
    def open_position(self, symbol: str, direction: str, entry_price: float,
                     quantity: float, stop_loss: float, take_profit: float,
                     strategy: str, confidence: float):
        """Ouvre une nouvelle position"""
        position = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'current_price': entry_price,
            'quantity': quantity,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'strategy': strategy,
            'confidence': confidence,
            'entry_time': datetime.now(),
            'unrealized_pnl': 0.0,
            'unrealized_pnl_pct': 0.0,
            'trailing_pct': 0.03,  # 3% trailing
            'trailing_stop': True
        }
        
        self.open_positions[symbol] = position
        
        logger.info(f"✅ Position opened: {direction} {quantity:.4f} {symbol} @ ${entry_price:.2f}")
        logger.info(f"   Stop Loss: ${stop_loss:.2f} | Take Profit: ${take_profit:.2f}")
    
    def close_position(self, symbol: str, exit_price: float, reason: str) -> Dict:
        """Ferme une position"""
        if symbol not in self.open_positions:
            logger.warning(f"⚠️ Cannot close position: {symbol} not found")
            return {}
        
        position = self.open_positions[symbol]
        
        # Calculate final P&L
        if position['direction'] == 'BUY':
            pnl = (exit_price - position['entry_price']) * position['quantity']
            pnl_pct = (exit_price / position['entry_price'] - 1) * 100
        else:
            pnl = (position['entry_price'] - exit_price) * position['quantity']
            pnl_pct = (1 - exit_price / position['entry_price']) * 100
        
        # Update capital
        self.current_capital += pnl
        self.total_pnl += pnl
        
        # Update peak
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Update equity curve
        self.equity_curve.append(self.current_capital)
        
        # Update daily P&L
        today = datetime.now().date().isoformat()
        if today not in self.daily_pnl:
            self.daily_pnl[today] = 0.0
        self.daily_pnl[today] += pnl
        
        # Update trade statistics
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Prepare result
        result = {
            'symbol': symbol,
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': reason,
            'strategy': position['strategy'],
            'hold_time': (datetime.now() - position['entry_time']).total_seconds() / 3600,  # hours
            'new_capital': self.current_capital
        }
        
        # Remove from open positions
        del self.open_positions[symbol]
        
        logger.info(f"🔚 Position closed: {symbol} | P&L: ${pnl:.2f} ({pnl_pct:+.2f}%) | {reason}")
        logger.info(f"   New capital: ${self.current_capital:.2f} (Total P&L: ${self.total_pnl:.2f})")
        
        return result
    
    def should_close_position(self, symbol: str) -> Tuple[bool, str]:
        """Vérifie si une position doit être fermée"""
        if symbol not in self.open_positions:
            return False, ""
        
        position = self.open_positions[symbol]
        current_price = position['current_price']
        
        # Check stop loss
        if position['direction'] == 'BUY':
            if current_price <= position['stop_loss']:
                return True, "Stop Loss"
        else:
            if current_price >= position['stop_loss']:
                return True, "Stop Loss"
        
        # Check take profit
        if position['direction'] == 'BUY':
            if current_price >= position['take_profit']:
                return True, "Take Profit"
        else:
            if current_price <= position['take_profit']:
                return True, "Take Profit"
        
        return False, ""
    
    def get_risk_metrics(self) -> Dict:
        """Calcule les métriques de risque actuelles"""
        # Portfolio heat
        _, portfolio_heat = self.check_portfolio_heat()
        
        # Drawdown
        _, current_drawdown = self.check_drawdown()
        
        # Win rate
        _, win_rate = self.check_win_rate()
        
        # Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = np.diff(list(self.equity_curve)) / list(self.equity_curve)[:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
        
        # Sortino ratio (only downside deviation)
        if len(self.equity_curve) > 1:
            returns = np.diff(list(self.equity_curve)) / list(self.equity_curve)[:-1]
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
            else:
                sortino = sharpe
        else:
            sortino = 0.0
        
        # Max historical drawdown
        equity_array = np.array(list(self.equity_curve))
        cummax = np.maximum.accumulate(equity_array)
        drawdowns = (cummax - equity_array) / cummax * 100
        max_historical_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        metrics = {
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'total_pnl': self.total_pnl,
            'total_return_pct': (self.current_capital / self.initial_capital - 1) * 100,
            'portfolio_heat': portfolio_heat,
            'current_drawdown': current_drawdown,
            'max_historical_drawdown': max_historical_drawdown,
            'win_rate': win_rate * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'open_positions': len(self.open_positions),
            'avg_win': self._calculate_avg_win(),
            'avg_loss': self._calculate_avg_loss(),
            'profit_factor': self._calculate_profit_factor()
        }
        
        # Store in history
        self.risk_metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        return metrics
    
    def _calculate_avg_win(self) -> float:
        """Calcule le gain moyen"""
        if self.winning_trades == 0:
            return 0.0
        # Simplified - in real version would track actual wins
        return abs(self.total_pnl) / max(self.winning_trades, 1) if self.total_pnl > 0 else 0.0
    
    def _calculate_avg_loss(self) -> float:
        """Calcule la perte moyenne"""
        if self.losing_trades == 0:
            return 0.0
        # Simplified - in real version would track actual losses
        return abs(self.total_pnl) / max(self.losing_trades, 1) if self.total_pnl < 0 else 0.0
    
    def _calculate_profit_factor(self) -> float:
        """Calcule le profit factor"""
        avg_win = self._calculate_avg_win()
        avg_loss = self._calculate_avg_loss()
        
        if avg_loss == 0:
            return float('inf') if avg_win > 0 else 0.0
        
        return (avg_win * self.winning_trades) / (avg_loss * self.losing_trades) if self.losing_trades > 0 else float('inf')
    
    def get_position_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calcule la corrélation entre deux positions"""
        # Simplified - in real version would use actual price history
        # For now, return random low correlation
        return np.random.uniform(-0.3, 0.3)
