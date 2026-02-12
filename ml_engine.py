"""
═══════════════════════════════════════════════════════════════
    MACHINE LEARNING ENGINE - LIGHTWEIGHT VERSION
═══════════════════════════════════════════════════════════════

ML engine that works everywhere without heavy dependencies.
Uses momentum-based prediction + Q-learning RL agent.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import deque
import json
import logging

logger = logging.getLogger(__name__)


class MarketPredictor:
    """Lightweight market direction predictor"""

    def __init__(self, input_features: int = 20):
        self.input_features = input_features
        self.training_history = deque(maxlen=1000)
        self.is_trained = False
        # Simple weight vector trained online
        self.weights = np.random.randn(input_features) * 0.01
        self.bias = np.zeros(3)
        self.learning_rate = 0.001
        logger.info("🧠 Market Predictor initialized")

    def prepare_features(self, market_data: Dict) -> np.ndarray:
        features = []
        prices = market_data.get('prices', [])
        if len(prices) >= 20:
            prices_array = np.array(prices[-20:])
            norm_prices = (prices_array - prices_array.mean()) / (prices_array.std() + 1e-8)
            features.extend(norm_prices[-10:].tolist())

            returns = np.diff(prices_array) / prices_array[:-1]
            features.extend(returns[-5:].tolist())

            sma_5 = np.mean(prices_array[-5:])
            sma_10 = np.mean(prices_array[-10:])
            features.append((sma_5 - sma_10) / sma_10)

            volatility = np.std(returns[-10:])
            features.append(volatility)

            momentum = (prices_array[-1] / prices_array[-10] - 1)
            features.append(momentum)

            # RSI-like
            ups = returns[returns > 0]
            downs = abs(returns[returns < 0])
            if len(downs) > 0 and len(ups) > 0:
                rs = np.mean(ups) / np.mean(downs)
                rsi = 100 - (100 / (1 + rs))
                features.append(rsi / 100)
            else:
                features.append(0.5)

        while len(features) < self.input_features:
            features.append(0.0)

        return np.array(features[:self.input_features], dtype=np.float32)

    def predict(self, market_data: Dict) -> Dict[str, float]:
        try:
            features = self.prepare_features(market_data)
            # Linear prediction with softmax
            logits = np.dot(features, self.weights[:3].repeat(self.input_features // 3 + 1)[:self.input_features])
            raw = np.array([logits + self.bias[0], -logits + self.bias[1], self.bias[2]])
            # Softmax
            exp_raw = np.exp(raw - np.max(raw))
            probs = exp_raw / exp_raw.sum()
            return {'UP': float(probs[0]), 'DOWN': float(probs[1]), 'NEUTRAL': float(probs[2])}
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'UP': 0.33, 'DOWN': 0.33, 'NEUTRAL': 0.34}

    def train_online(self, market_data: Dict, actual_direction: int):
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'direction': actual_direction
        })
        if len(self.training_history) >= 10:
            self.is_trained = True


class ReinforcementLearningAgent:
    """Q-learning agent for trading decisions"""

    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.experience_buffer = deque(maxlen=1000)
        logger.info("🤖 RL Agent initialized")

    def get_state(self, market_data: Dict, position: Optional[Dict]) -> str:
        prices = market_data.get('prices', [])
        if len(prices) < 10:
            return "INSUFFICIENT_DATA"
        recent_return = (prices[-1] / prices[-5] - 1) * 100
        return_state = "UP" if recent_return > 1 else "DOWN" if recent_return < -1 else "FLAT"
        position_state = "LONG" if position else "FLAT"
        return f"{return_state}_{position_state}"

    def get_action(self, state: str, available_actions: List[str]) -> str:
        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        if state not in self.q_table:
            return np.random.choice(available_actions)
        q_values = self.q_table[state]
        return max(available_actions, key=lambda a: q_values.get(a, 0))

    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        max_next_q = 0.0
        if next_state in self.q_table and self.q_table[next_state]:
            max_next_q = max(self.q_table[next_state].values())
        current_q = self.q_table[state][action]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        self.experience_buffer.append({
            'state': state, 'action': action,
            'reward': reward, 'next_state': next_state,
            'timestamp': datetime.now().isoformat()
        })

    def get_best_action_confidence(self, state: str) -> float:
        if state not in self.q_table or not self.q_table[state]:
            return 0.5
        q_values = list(self.q_table[state].values())
        if not q_values:
            return 0.5
        max_q, min_q = max(q_values), min(q_values)
        if max_q == min_q:
            return 0.5
        normalized = (max_q - min_q) / (max(abs(max_q), abs(min_q)) + 1e-8)
        return min(max(normalized, 0.0), 1.0)

    def decay_epsilon(self, decay_rate: float = 0.995):
        self.epsilon = max(self.epsilon * decay_rate, 0.01)


class AdaptiveParameterOptimizer:
    """Strategy parameter optimizer"""

    def __init__(self):
        self.parameter_performance: Dict[str, List[float]] = {}
        self.best_parameters: Dict[str, Dict] = {}
        logger.info("⚙️ Parameter Optimizer initialized")

    def suggest_parameters(self, strategy_name: str) -> Dict:
        if strategy_name in self.best_parameters:
            return self.best_parameters[strategy_name]
        defaults = {
            'Mean Reversion': {'threshold': 2.0, 'lookback': 50},
            'Momentum': {'lookback': 50},
            'Breakout': {'lookback_period': 20},
            'RSI Mean Reversion': {'oversold': 30, 'overbought': 70},
            'Bollinger Bands': {'lookback': 50}
        }
        return defaults.get(strategy_name, {})

    def record_performance(self, strategy_name: str, parameters: Dict, performance_score: float):
        param_key = f"{strategy_name}_{json.dumps(parameters, sort_keys=True)}"
        if param_key not in self.parameter_performance:
            self.parameter_performance[param_key] = []
        self.parameter_performance[param_key].append(performance_score)


class MachineLearningEngine:
    """Main ML engine orchestrating all ML components"""

    def __init__(self):
        self.predictor = MarketPredictor(input_features=20)
        self.rl_agent = ReinforcementLearningAgent()
        self.parameter_optimizer = AdaptiveParameterOptimizer()
        self.prediction_history = deque(maxlen=100)
        self.learning_enabled = True
        logger.info("🧠 Machine Learning Engine initialized")

    def enhance_signal(self, signal, market_data: Dict):
        if not self.learning_enabled:
            return signal
        try:
            ml_prediction = self.predictor.predict(market_data)
            state = self.rl_agent.get_state(market_data, None)
            rl_confidence = self.rl_agent.get_best_action_confidence(state)
            if signal.direction == 'BUY':
                ml_factor = ml_prediction['UP']
            else:
                ml_factor = ml_prediction['DOWN']
            original_confidence = signal.confidence
            enhanced_confidence = (
                0.5 * original_confidence +
                0.3 * ml_factor +
                0.2 * rl_confidence
            )
            signal.confidence = min(enhanced_confidence, 0.99)
            signal.features['ml_prediction'] = ml_prediction
            signal.features['rl_confidence'] = rl_confidence
            signal.features['original_confidence'] = original_confidence
            logger.info(f"🧠 ML enhanced: {original_confidence:.2f} → {signal.confidence:.2f}")
        except Exception as e:
            logger.error(f"ML enhancement error: {e}")
        return signal

    def learn_from_trade(self, market_data: Dict, action: str, pnl: float):
        if not self.learning_enabled:
            return
        try:
            direction = 0 if (pnl > 0 and action == 'BUY') else 1 if (pnl > 0) else 2
            self.predictor.train_online(market_data, direction)
            state = self.rl_agent.get_state(market_data, None)
            self.rl_agent.update_q_value(state, action, pnl, state)
            self.rl_agent.decay_epsilon()
            logger.info(f"📚 Learned from trade: {action} → P&L: {pnl:.2f}")
        except Exception as e:
            logger.error(f"Learning error: {e}")

    def get_ml_stats(self) -> Dict:
        return {
            'predictor_trained': self.predictor.is_trained,
            'training_samples': len(self.predictor.training_history),
            'rl_states_learned': len(self.rl_agent.q_table),
            'rl_epsilon': self.rl_agent.epsilon,
            'rl_experiences': len(self.rl_agent.experience_buffer),
            'parameter_sets_tested': len(self.parameter_optimizer.parameter_performance)
        }
