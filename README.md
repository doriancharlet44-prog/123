# 🚀 Professional Trading Platform

Automated trading bot with IBKR integration, multi-strategy engine, ML predictions, and German tax compliance.

## Features

- **5 Trading Strategies**: Mean Reversion, Momentum, Breakout, RSI, Bollinger Bands
- **ML Engine**: Momentum predictor + Q-learning reinforcement learning
- **Risk Management**: Kelly Criterion position sizing, drawdown protection, daily loss limits
- **German Tax (26.375%)**: Automatic KapESt + Soli calculation, CSV export for Finanzamt
- **Interactive Brokers**: Live/paper trading via ib-insync (auto-falls back to demo mode)
- **Professional Dashboard**: Real-time stats, equity curve, positions, trade history

## Files

| File | Description |
|------|-------------|
| `app.py` | Flask web app + full HTML frontend + API routes |
| `trading_bot.py` | Main bot engine, IBKR connector, trading loop |
| `strategies.py` | 5 trading strategies with technical indicators |
| `ml_engine.py` | Market predictor, RL agent, parameter optimizer |
| `risk_manager.py` | Kelly Criterion, drawdown protection, position sizing |
| `database.py` | SQLite database with 8 tables |
| `requirements.txt` | Python dependencies |
| `Procfile` | Railway/Heroku start command |
| `railway.toml` | Railway deployment config |
| `runtime.txt` | Python version |

## Deploy to Railway

1. Push all files to a **GitHub repository**
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
3. Railway auto-detects Python, installs deps, and starts gunicorn
4. Your app will be live at `https://your-app.up.railway.app`

No environment variables needed — the app works out-of-the-box in demo mode.

## Local Development

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## Live Trading (Optional)

1. Install [TWS](https://www.interactivebrokers.com/en/trading/tws.php) or IB Gateway
2. Enable API access in TWS: Edit → Global Configuration → API → Settings
3. Install ib-insync: `pip install ib-insync`
4. Update host/port in the Config tab
5. Start the bot — it will auto-connect to IBKR

## Tax Report

The platform calculates German capital gains tax (26.375% = 25% KapESt + 5.5% Soli) and lets you export a CSV for your Finanzamt declaration (Anlage KAP).
