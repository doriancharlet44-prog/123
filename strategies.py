"""
═══════════════════════════════════════════════════════════════
    STRATEGY ENGINE - PROFESSIONAL MULTI-STRATEGY SYSTEM
═══════════════════════════════════════════════════════════════

Système de stratégies de trading modulaires et évolutives
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Représente un signal de trading"""
    symbol: str
    direction: str  # 'BUY' or 'SELL'
    confidence: float  # 0-1
    strategy: str
    entry_price: float
    stop_loss: float
    take_profit: float
    features: Dict
    timestamp: datetime
    reason: str


class BaseStrategy(ABC):
    """Classe de base pour toutes les stratégies"""
    
    def __init__(self, name: str, lookback: int = 50):
        self.name = name
        self.lookback = lookback
        self.price_history = {}
        self.enabled = True
        self.performance_score = 1.0
        self.trade_count = 0
        self.win_count = 0
    
    @abstractmethod
    def analyze(self, symbol: str, current_price: float, market_data: Dict) -> Optional[Signal]:
        """Analyse le marché et génère un signal"""
        pass
    
    def update_price_history(self, symbol: str, price: float):
        """Met à jour l'historique des prix"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.lookback)
        self.price_history[symbol].append(price)
    
    def get_prices(self, symbol: str) -> List[float]:
        """Récupère l'historique des prix"""
        if symbol not in self.price_history:
            return []
        return list(self.price_history[symbol])
    
    def calculate_technical_indicators(self, prices: List[float]) -> Dict:
        """Calcule les indicateurs techniques standards"""
        if len(prices) < 20:
            return {}
        
        prices_array = np.array(prices)
        
        indicators = {}
        
        # Moving Averages
        indicators['sma_20'] = np.mean(prices_array[-20:])
        indicators['sma_50'] = np.mean(prices_array[-50:]) if len(prices) >= 50 else indicators['sma_20']
        
        # EMA
        indicators['ema_12'] = self._calculate_ema(prices_array, 12)
        indicators['ema_26'] = self._calculate_ema(prices_array, 26)
        
        # MACD
        indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
        macd_values = []
        for i in range(9, len(prices)):
            macd_val = self._calculate_ema(prices_array[i-8:i+1], 12) - self._calculate_ema(prices_array[i-8:i+1], 26)
            macd_values.append(macd_val)
        indicators['macd_signal'] = np.mean(macd_values[-9:]) if len(macd_values) >= 9 else indicators['macd']
        
        # RSI
        indicators['rsi'] = self._calculate_rsi(prices_array)
        
        # Bollinger Bands
        sma_20 = indicators['sma_20']
        std_20 = np.std(prices_array[-20:])
        indicators['bb_upper'] = sma_20 + (2 * std_20)
        indicators['bb_lower'] = sma_20 - (2 * std_20)
        
        # ATR (Average True Range)
        indicators['atr'] = self._calculate_atr(prices_array)
        
        # Stochastic
        indicators['stoch_k'], indicators['stoch_d'] = self._calculate_stochastic(prices_array)
        
        return indicators
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calcule l'EMA"""
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calcule le RSI"""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, prices: np.ndarray, period: int = 14) -> float:
        """Calcule l'ATR"""
        if len(prices) < period + 1:
            return prices[-1] * 0.02
        
        high_low = prices[-period:].max() - prices[-period:].min()
        return high_low / period
    
    def _calculate_stochastic(self, prices: np.ndarray, period: int = 14) -> Tuple[float, float]:
        """Calcule le Stochastic Oscillator"""
        if len(prices) < period:
            return 50, 50
        
        low_min = np.min(prices[-period:])
        high_max = np.max(prices[-period:])
        
        if high_max == low_min:
            return 50, 50
        
        k = ((prices[-1] - low_min) / (high_max - low_min)) * 100
        
        # D is 3-period SMA of K (simplified here)
        d = k  # In real implementation, would calculate SMA of last 3 K values
        
        return k, d
    
    def update_performance(self, trade_won: bool):
        """Met à jour les métriques de performance"""
        self.trade_count += 1
        if trade_won:
            self.win_count += 1
        
        if self.trade_count > 0:
            win_rate = self.win_count / self.trade_count
            # Update performance score (weighted moving average)
            self.performance_score = 0.7 * self.performance_score + 0.3 * win_rate


class MeanReversionStrategy(BaseStrategy):
    """Stratégie de retour à la moyenne"""
    
    def __init__(self, threshold: float = 2.0):
        super().__init__("Mean Reversion", lookback=50)
        self.threshold = threshold
    
    def analyze(self, symbol: str, current_price: float, market_data: Dict) -> Optional[Signal]:
        """Analyse pour mean reversion"""
        prices = self.get_prices(symbol)
        
        if len(prices) < 20:
            return None
        
        # Calculate Z-score
        mean = np.mean(prices)
        std = np.std(prices)
        
        if std == 0:
            return None
        
        z_score = (current_price - mean) / std
        
        # Get technical indicators
        indicators = self.calculate_technical_indicators(prices)
        
        # Mean reversion signal
        if z_score < -self.threshold:
            # Oversold - BUY signal
            confidence = min(abs(z_score) / 3, 1.0)  # Higher z-score = higher confidence
            
            # Additional confirmation from RSI
            if indicators.get('rsi', 50) < 30:
                confidence = min(confidence * 1.2, 1.0)
            
            stop_loss = current_price * 0.98
            take_profit = mean  # Target: return to mean
            
            return Signal(
                symbol=symbol,
                direction='BUY',
                confidence=confidence,
                strategy=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                features={
                    'z_score': z_score,
                    'rsi': indicators.get('rsi', 50),
                    'mean': mean,
                    'std': std
                },
                timestamp=datetime.now(),
                reason=f"Oversold (Z-score: {z_score:.2f})"
            )
        
        elif z_score > self.threshold:
            # Overbought - SELL signal
            confidence = min(abs(z_score) / 3, 1.0)
            
            if indicators.get('rsi', 50) > 70:
                confidence = min(confidence * 1.2, 1.0)
            
            stop_loss = current_price * 1.02
            take_profit = mean
            
            return Signal(
                symbol=symbol,
                direction='SELL',
                confidence=confidence,
                strategy=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                features={
                    'z_score': z_score,
                    'rsi': indicators.get('rsi', 50),
                    'mean': mean,
                    'std': std
                },
                timestamp=datetime.now(),
                reason=f"Overbought (Z-score: {z_score:.2f})"
            )
        
        return None


class MomentumStrategy(BaseStrategy):
    """Stratégie de momentum / trend following"""
    
    def __init__(self):
        super().__init__("Momentum", lookback=50)
    
    def analyze(self, symbol: str, current_price: float, market_data: Dict) -> Optional[Signal]:
        """Analyse pour momentum trading"""
        prices = self.get_prices(symbol)
        
        if len(prices) < 50:
            return None
        
        indicators = self.calculate_technical_indicators(prices)
        
        # Momentum signals
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        sma_20 = indicators['sma_20']
        sma_50 = indicators['sma_50']
        
        # Bullish momentum
        if (macd > macd_signal and 
            sma_20 > sma_50 and 
            current_price > sma_20):
            
            # Calculate momentum strength
            momentum = (current_price / prices[-20] - 1) * 100
            confidence = min(abs(momentum) / 10, 0.9)
            
            atr = indicators.get('atr', current_price * 0.02)
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (3 * atr)
            
            return Signal(
                symbol=symbol,
                direction='BUY',
                confidence=confidence,
                strategy=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                features={
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'momentum': momentum
                },
                timestamp=datetime.now(),
                reason=f"Bullish Momentum ({momentum:+.1f}%)"
            )
        
        # Bearish momentum
        elif (macd < macd_signal and 
              sma_20 < sma_50 and 
              current_price < sma_20):
            
            momentum = (current_price / prices[-20] - 1) * 100
            confidence = min(abs(momentum) / 10, 0.9)
            
            atr = indicators.get('atr', current_price * 0.02)
            stop_loss = current_price + (2 * atr)
            take_profit = current_price - (3 * atr)
            
            return Signal(
                symbol=symbol,
                direction='SELL',
                confidence=confidence,
                strategy=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                features={
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'momentum': momentum
                },
                timestamp=datetime.now(),
                reason=f"Bearish Momentum ({momentum:+.1f}%)"
            )
        
        return None


class BreakoutStrategy(BaseStrategy):
    """Stratégie de breakout / cassure de niveaux"""
    
    def __init__(self, lookback_period: int = 20):
        super().__init__("Breakout", lookback=lookback_period)
        self.lookback_period = lookback_period
    
    def analyze(self, symbol: str, current_price: float, market_data: Dict) -> Optional[Signal]:
        """Analyse pour breakout trading"""
        prices = self.get_prices(symbol)
        
        if len(prices) < self.lookback_period:
            return None
        
        prices_array = np.array(prices)
        
        # Calculate support and resistance
        resistance = np.max(prices_array[-self.lookback_period:])
        support = np.min(prices_array[-self.lookback_period:])
        
        indicators = self.calculate_technical_indicators(prices)
        atr = indicators.get('atr', current_price * 0.02)
        
        # Breakout above resistance
        if current_price > resistance * 1.002:  # 0.2% above resistance
            # Confirm with volume and momentum
            confidence = 0.7
            
            # Additional confirmation from RSI
            rsi = indicators.get('rsi', 50)
            if 40 < rsi < 70:  # Not overbought
                confidence = min(confidence * 1.2, 0.95)
            
            stop_loss = resistance  # Previous resistance becomes support
            take_profit = current_price + (current_price - resistance) * 2  # 2:1 reward
            
            return Signal(
                symbol=symbol,
                direction='BUY',
                confidence=confidence,
                strategy=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                features={
                    'resistance': resistance,
                    'support': support,
                    'breakout_strength': (current_price / resistance - 1) * 100,
                    'rsi': rsi
                },
                timestamp=datetime.now(),
                reason=f"Resistance Breakout (${resistance:.2f})"
            )
        
        # Breakdown below support
        elif current_price < support * 0.998:  # 0.2% below support
            confidence = 0.7
            
            rsi = indicators.get('rsi', 50)
            if 30 < rsi < 60:  # Not oversold
                confidence = min(confidence * 1.2, 0.95)
            
            stop_loss = support  # Previous support becomes resistance
            take_profit = current_price - (support - current_price) * 2
            
            return Signal(
                symbol=symbol,
                direction='SELL',
                confidence=confidence,
                strategy=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                features={
                    'resistance': resistance,
                    'support': support,
                    'breakdown_strength': (1 - current_price / support) * 100,
                    'rsi': rsi
                },
                timestamp=datetime.now(),
                reason=f"Support Breakdown (${support:.2f})"
            )
        
        return None


class RSIMeanReversionStrategy(BaseStrategy):
    """Stratégie RSI Mean Reversion"""
    
    def __init__(self, oversold: int = 30, overbought: int = 70):
        super().__init__("RSI Mean Reversion", lookback=50)
        self.oversold = oversold
        self.overbought = overbought
    
    def analyze(self, symbol: str, current_price: float, market_data: Dict) -> Optional[Signal]:
        """Analyse basée sur RSI"""
        prices = self.get_prices(symbol)
        
        if len(prices) < 30:
            return None
        
        indicators = self.calculate_technical_indicators(prices)
        rsi = indicators.get('rsi', 50)
        
        atr = indicators.get('atr', current_price * 0.02)
        
        # Oversold condition
        if rsi < self.oversold:
            # More oversold = higher confidence
            confidence = (self.oversold - rsi) / self.oversold
            confidence = min(confidence, 0.95)
            
            # Additional confirmation from Stochastic
            stoch_k = indicators.get('stoch_k', 50)
            if stoch_k < 20:
                confidence = min(confidence * 1.15, 0.95)
            
            stop_loss = current_price - (2 * atr)
            take_profit = current_price + (3 * atr)
            
            return Signal(
                symbol=symbol,
                direction='BUY',
                confidence=confidence,
                strategy=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                features={
                    'rsi': rsi,
                    'stoch_k': stoch_k,
                    'atr': atr
                },
                timestamp=datetime.now(),
                reason=f"RSI Oversold ({rsi:.1f})"
            )
        
        # Overbought condition
        elif rsi > self.overbought:
            confidence = (rsi - self.overbought) / (100 - self.overbought)
            confidence = min(confidence, 0.95)
            
            stoch_k = indicators.get('stoch_k', 50)
            if stoch_k > 80:
                confidence = min(confidence * 1.15, 0.95)
            
            stop_loss = current_price + (2 * atr)
            take_profit = current_price - (3 * atr)
            
            return Signal(
                symbol=symbol,
                direction='SELL',
                confidence=confidence,
                strategy=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                features={
                    'rsi': rsi,
                    'stoch_k': stoch_k,
                    'atr': atr
                },
                timestamp=datetime.now(),
                reason=f"RSI Overbought ({rsi:.1f})"
            )
        
        return None


class BollingerBandsStrategy(BaseStrategy):
    """Stratégie Bollinger Bands"""
    
    def __init__(self):
        super().__init__("Bollinger Bands", lookback=50)
    
    def analyze(self, symbol: str, current_price: float, market_data: Dict) -> Optional[Signal]:
        """Analyse avec Bollinger Bands"""
        prices = self.get_prices(symbol)
        
        if len(prices) < 20:
            return None
        
        indicators = self.calculate_technical_indicators(prices)
        
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        sma_20 = indicators['sma_20']
        
        atr = indicators.get('atr', current_price * 0.02)
        
        # Price touches lower band - potential bounce
        if current_price <= bb_lower * 1.001:  # Within 0.1% of lower band
            # Calculate distance from middle band
            distance = (sma_20 - current_price) / sma_20
            confidence = min(distance * 10, 0.9)
            
            # Additional confirmation
            rsi = indicators.get('rsi', 50)
            if rsi < 40:
                confidence = min(confidence * 1.2, 0.95)
            
            stop_loss = current_price - atr
            take_profit = sma_20  # Target middle band
            
            return Signal(
                symbol=symbol,
                direction='BUY',
                confidence=confidence,
                strategy=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                features={
                    'bb_upper': bb_upper,
                    'bb_lower': bb_lower,
                    'bb_middle': sma_20,
                    'distance_from_middle': distance * 100,
                    'rsi': rsi
                },
                timestamp=datetime.now(),
                reason="Lower BB Touch"
            )
        
        # Price touches upper band - potential reversal
        elif current_price >= bb_upper * 0.999:  # Within 0.1% of upper band
            distance = (current_price - sma_20) / sma_20
            confidence = min(distance * 10, 0.9)
            
            rsi = indicators.get('rsi', 50)
            if rsi > 60:
                confidence = min(confidence * 1.2, 0.95)
            
            stop_loss = current_price + atr
            take_profit = sma_20
            
            return Signal(
                symbol=symbol,
                direction='SELL',
                confidence=confidence,
                strategy=self.name,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                features={
                    'bb_upper': bb_upper,
                    'bb_lower': bb_lower,
                    'bb_middle': sma_20,
                    'distance_from_middle': distance * 100,
                    'rsi': rsi
                },
                timestamp=datetime.now(),
                reason="Upper BB Touch"
            )
        
        return None


class StrategyManager:
    """Gestionnaire de stratégies multiples"""
    
    def __init__(self):
        self.strategies: List[BaseStrategy] = []
        self.initialize_strategies()
    
    def initialize_strategies(self):
        """Initialise toutes les stratégies"""
        self.strategies = [
            MeanReversionStrategy(threshold=2.0),
            MomentumStrategy(),
            BreakoutStrategy(lookback_period=20),
            RSIMeanReversionStrategy(oversold=30, overbought=70),
            BollingerBandsStrategy()
        ]
        
        logger.info(f"✅ {len(self.strategies)} strategies initialized")
    
    def update_all_prices(self, symbol: str, price: float):
        """Met à jour le prix pour toutes les stratégies"""
        for strategy in self.strategies:
            if strategy.enabled:
                strategy.update_price_history(symbol, price)
    
    def get_best_signal(self, symbol: str, current_price: float, market_data: Dict) -> Optional[Signal]:
        """Récupère le meilleur signal parmi toutes les stratégies"""
        signals = []
        
        for strategy in self.strategies:
            if not strategy.enabled:
                continue
            
            try:
                signal = strategy.analyze(symbol, current_price, market_data)
                if signal:
                    # Weight signal by strategy performance
                    signal.confidence *= strategy.performance_score
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error in strategy {strategy.name}: {e}")
        
        if not signals:
            return None
        
        # Return signal with highest confidence
        best_signal = max(signals, key=lambda s: s.confidence)
        
        logger.info(f"📊 Best signal: {best_signal.strategy} - {best_signal.direction} "
                   f"({best_signal.confidence:.1%} confidence)")
        
        return best_signal
    
    def get_all_signals(self, symbol: str, current_price: float, market_data: Dict) -> List[Signal]:
        """Récupère tous les signaux de toutes les stratégies"""
        signals = []
        
        for strategy in self.strategies:
            if not strategy.enabled:
                continue
            
            try:
                signal = strategy.analyze(symbol, current_price, market_data)
                if signal:
                    signal.confidence *= strategy.performance_score
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error in strategy {strategy.name}: {e}")
        
        return signals
    
    def update_strategy_performance(self, strategy_name: str, trade_won: bool):
        """Met à jour la performance d'une stratégie"""
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                strategy.update_performance(trade_won)
                logger.info(f"📈 {strategy_name} performance: {strategy.performance_score:.2f} "
                          f"({strategy.win_count}/{strategy.trade_count} wins)")
                break
    
    def get_strategies_status(self) -> List[Dict]:
        """Récupère le status de toutes les stratégies"""
        return [{
            'name': s.name,
            'enabled': s.enabled,
            'performance_score': s.performance_score,
            'trades': s.trade_count,
            'wins': s.win_count,
            'win_rate': (s.win_count / s.trade_count * 100) if s.trade_count > 0 else 0
        } for s in self.strategies]
    
    def enable_strategy(self, strategy_name: str):
        """Active une stratégie"""
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                strategy.enabled = True
                logger.info(f"✅ Strategy '{strategy_name}' enabled")
                break
    
    def disable_strategy(self, strategy_name: str):
        """Désactive une stratégie"""
        for strategy in self.strategies:
            if strategy.name == strategy_name:
                strategy.enabled = False
                logger.info(f"🛑 Strategy '{strategy_name}' disabled")
                break
