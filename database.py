"""
═══════════════════════════════════════════════════════════════
    DATABASE MODELS - PROFESSIONAL TRADING PLATFORM
═══════════════════════════════════════════════════════════════
"""

from datetime import datetime
from typing import Optional, List, Dict
import json
import sqlite3
import os
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Gestionnaire de base de données professionnel"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Use /tmp for Railway (ephemeral storage) or local
            data_dir = os.environ.get("DATA_DIR", "/tmp")
            db_path = os.path.join(data_dir, "trading_platform.db")
        self.db_path = db_path
        self.init_database()
        logger.info(f"📦 Database initialized at {self.db_path}")

    def get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def init_database(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                strategy TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                exit_time TIMESTAMP,
                exit_price REAL,
                pnl REAL,
                pnl_pct REAL,
                commission REAL DEFAULT 0,
                slippage REAL DEFAULT 0,
                stop_loss REAL,
                take_profit REAL,
                confidence REAL,
                features TEXT,
                status TEXT DEFAULT 'OPEN',
                exit_reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                date DATE NOT NULL,
                trades_count INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                sharpe_ratio REAL,
                max_drawdown REAL,
                avg_trade_duration INTEGER,
                parameters TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(strategy_name, date)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                indicators TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                version INTEGER NOT NULL,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                training_samples INTEGER,
                features_used TEXT,
                hyperparameters TEXT,
                trained_at TIMESTAMP,
                is_active BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                total_exposure REAL,
                portfolio_heat REAL,
                max_drawdown REAL,
                current_drawdown REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                var_95 REAL,
                position_count INTEGER,
                capital REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS configuration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                type TEXT NOT NULL,
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                symbol TEXT,
                triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                acknowledged BOOLEAN DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                strategy TEXT NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                initial_capital REAL NOT NULL,
                final_capital REAL,
                total_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                total_trades INTEGER,
                parameters TEXT,
                results TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data(symbol, timestamp)")

        conn.commit()
        conn.close()

    # ─── Trade Operations ───────────────────────────────────────

    def insert_trade(self, trade_data: Dict) -> int:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO trades (
                symbol, strategy, direction, entry_time, entry_price, quantity,
                stop_loss, take_profit, confidence, features, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_data['symbol'],
            trade_data['strategy'],
            trade_data['direction'],
            trade_data['entry_time'],
            trade_data['entry_price'],
            trade_data['quantity'],
            trade_data.get('stop_loss'),
            trade_data.get('take_profit'),
            trade_data.get('confidence'),
            json.dumps(trade_data.get('features', {})),
            'OPEN'
        ))
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return trade_id

    def close_trade(self, trade_id: int, exit_price: float, exit_reason: str):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        trade = cursor.fetchone()
        if trade:
            if trade['direction'] == 'BUY':
                pnl = (exit_price - trade['entry_price']) * trade['quantity']
                pnl_pct = (exit_price / trade['entry_price'] - 1) * 100
            else:
                pnl = (trade['entry_price'] - exit_price) * trade['quantity']
                pnl_pct = (1 - exit_price / trade['entry_price']) * 100
            cursor.execute("""
                UPDATE trades
                SET exit_time = ?, exit_price = ?, pnl = ?, pnl_pct = ?,
                    status = 'CLOSED', exit_reason = ?
                WHERE id = ?
            """, (datetime.now(), exit_price, pnl, pnl_pct, exit_reason, trade_id))
            conn.commit()
        conn.close()

    def get_open_trades(self) -> List[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM trades WHERE status = 'OPEN' ORDER BY entry_time DESC")
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return trades

    def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM trades
            WHERE status = 'CLOSED'
            ORDER BY exit_time DESC
            LIMIT ?
        """, (limit,))
        trades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return trades

    # ─── Configuration ──────────────────────────────────────────

    def set_config(self, key: str, value, config_type: str = "str", description: str = None):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO configuration (key, value, type, description, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (key, json.dumps(value), config_type, description, datetime.now()))
        conn.commit()
        conn.close()

    def get_config(self, key: str, default=None):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT value, type FROM configuration WHERE key = ?", (key,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return json.loads(result['value'])
        return default

    def get_all_config(self) -> Dict:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT key, value, type, description FROM configuration")
        configs = {}
        for row in cursor.fetchall():
            configs[row['key']] = {
                'value': json.loads(row['value']),
                'type': row['type'],
                'description': row['description']
            }
        conn.close()
        return configs

    # ─── Alerts ─────────────────────────────────────────────────

    def create_alert(self, alert_type: str, severity: str, message: str, symbol: str = None):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alerts (alert_type, severity, message, symbol)
            VALUES (?, ?, ?, ?)
        """, (alert_type, severity, message, symbol))
        conn.commit()
        conn.close()

    def get_unacknowledged_alerts(self) -> List[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM alerts WHERE acknowledged = 0 ORDER BY triggered_at DESC")
        alerts = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return alerts


# Singleton
_db_instance = None

def get_db() -> DatabaseManager:
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
