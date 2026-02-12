"""
═══════════════════════════════════════════════════════════════
    PROFESSIONAL TRADING PLATFORM - FLASK APPLICATION
═══════════════════════════════════════════════════════════════
"""

from flask import Flask, jsonify, request, Response
try:
    from flask_cors import CORS
except ImportError:
    CORS = None
import logging
import os
from datetime import datetime
import csv
from io import StringIO
import traceback

# ─── App Setup ──────────────────────────────────────────────

app = Flask(__name__)
if CORS:
    CORS(app)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Lazy-init to avoid crashes at import time
_bot = None
_db = None

def get_bot_safe():
    global _bot
    if _bot is None:
        from trading_bot import get_bot
        _bot = get_bot()
    return _bot

def get_db_safe():
    global _db
    if _db is None:
        from database import get_db
        _db = get_db()
    return _db

GERMAN_TAX_RATE = 0.26375

# ═══════════════════════════════════════════════════════════════
#  FRONTEND — Single-Page Application
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return FRONTEND_HTML

# ═══════════════════════════════════════════════════════════════
#  API ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/api/bot/status')
def api_status():
    try:
        bot = get_bot_safe()
        return jsonify({'success': True, 'data': bot.get_status()})
    except Exception as e:
        logger.error(f"Status error: {e}")
        return jsonify({'success': True, 'data': _fallback_status()})

@app.route('/api/bot/start', methods=['POST'])
def api_start():
    try:
        bot = get_bot_safe()
        return jsonify({'success': bot.start()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bot/stop', methods=['POST'])
def api_stop():
    try:
        bot = get_bot_safe()
        return jsonify({'success': bot.stop()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bot/pause', methods=['POST'])
def api_pause():
    try:
        bot = get_bot_safe()
        bot.pause()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/bot/resume', methods=['POST'])
def api_resume():
    try:
        bot = get_bot_safe()
        bot.resume()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/strategies')
def api_strategies():
    try:
        bot = get_bot_safe()
        return jsonify({'success': True, 'data': bot.strategy_manager.get_strategies_status()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trades')
def api_trades():
    try:
        db = get_db_safe()
        return jsonify({'success': True, 'data': db.get_recent_trades(100)})
    except Exception as e:
        return jsonify({'success': True, 'data': []})

@app.route('/api/taxes')
def api_taxes():
    try:
        db = get_db_safe()
        trades = db.get_recent_trades(1000)
        profit = sum(t['pnl'] for t in trades if t.get('pnl') and t['pnl'] > 0)
        loss = sum(abs(t['pnl']) for t in trades if t.get('pnl') and t['pnl'] < 0)
        taxable = profit - loss
        tax = taxable * GERMAN_TAX_RATE if taxable > 0 else 0
        return jsonify({'success': True, 'data': {
            'total_tax': tax,
            'taxable_profit': taxable,
            'gross_profit': profit,
            'total_loss': loss
        }})
    except Exception as e:
        return jsonify({'success': True, 'data': {
            'total_tax': 0, 'taxable_profit': 0,
            'gross_profit': 0, 'total_loss': 0
        }})

@app.route('/api/taxes/export')
def api_tax_export():
    try:
        db = get_db_safe()
        trades = db.get_recent_trades(1000)
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Date', 'Symbol', 'Action', 'Qty', 'Entry', 'Exit', 'P&L', 'Tax (26.375%)'])
        for t in trades:
            if t.get('pnl'):
                tax = t['pnl'] * GERMAN_TAX_RATE if t['pnl'] > 0 else 0
                writer.writerow([
                    t['entry_time'], t['symbol'], t['direction'],
                    t['quantity'], t['entry_price'],
                    t.get('exit_price', ''), f"{t['pnl']:.2f}", f"{tax:.2f}"
                ])
        return Response(
            output.getvalue(), mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename=finanzamt_report_{datetime.now().year}.csv'}
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'GET':
        try:
            db = get_db_safe()
            return jsonify({'success': True, 'data': db.get_all_config()})
        except Exception:
            return jsonify({'success': True, 'data': {}})
    else:
        try:
            bot = get_bot_safe()
            bot.update_config(request.json)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/alerts')
def api_alerts():
    try:
        db = get_db_safe()
        return jsonify({'success': True, 'data': db.get_unacknowledged_alerts()})
    except Exception:
        return jsonify({'success': True, 'data': []})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


def _fallback_status():
    """Fallback status when bot isn't fully initialized"""
    return {
        'running': False, 'paused': False, 'connected': False,
        'simulation_mode': True, 'trade_count': 0, 'uptime': '0s',
        'current_capital': 500, 'peak_capital': 500, 'total_pnl': 0,
        'total_return_pct': 0, 'portfolio_heat': 0, 'current_drawdown': 0,
        'max_historical_drawdown': 0, 'win_rate': 0, 'total_trades': 0,
        'winning_trades': 0, 'losing_trades': 0, 'sharpe_ratio': 0,
        'sortino_ratio': 0, 'open_positions': 0, 'avg_win': 0,
        'avg_loss': 0, 'profit_factor': 0,
        'open_positions_list': [], 'strategies': [],
        'ml_stats': {
            'predictor_trained': False, 'training_samples': 0,
            'rl_states_learned': 0, 'rl_epsilon': 0.1,
            'rl_experiences': 0, 'parameter_sets_tested': 0
        }
    }


# ═══════════════════════════════════════════════════════════════
#  FULL HTML FRONTEND
# ═══════════════════════════════════════════════════════════════

FRONTEND_HTML = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Professional Trading Platform</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🚀</text></svg>">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0"></script>
<style>
:root {
    --bg:       #0b0f1e;
    --surface:  #141929;
    --surface2: #1c2237;
    --border:   rgba(255,255,255,.08);
    --text:     #e2e8f0;
    --muted:    #8892a7;
    --green:    #10b981;
    --red:      #ef4444;
    --blue:     #667eea;
    --purple:   #a78bfa;
    --amber:    #f59e0b;
    --radius:   14px;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: 'Inter', -apple-system, system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }

/* ── Header ─────────────────────────────────── */
.header {
    background: var(--surface);
    padding: 16px 28px;
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 100;
    backdrop-filter: blur(12px);
}
.header h1 {
    font-size: 22px; font-weight: 700;
    background: linear-gradient(135deg, var(--blue), var(--purple));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.header-right { display:flex; gap:10px; align-items:center; }
.badge {
    padding: 6px 14px; border-radius: 20px;
    font-size: 12px; font-weight: 600;
    display: inline-flex; align-items: center; gap: 5px;
}
.badge-on  { background: rgba(16,185,129,.15); color: var(--green); }
.badge-off { background: rgba(239,68,68,.15);  color: var(--red); }
.badge-warn{ background: rgba(245,158,11,.15); color: var(--amber); }

/* ── Layout ─────────────────────────────────── */
.container { max-width:1500px; margin:0 auto; padding:24px; }
.tabs {
    display:flex; gap:4px; margin-bottom:24px;
    background: var(--surface); border-radius: var(--radius); padding:4px;
    overflow-x: auto;
}
.tab {
    padding:12px 22px; cursor:pointer; border-radius:10px;
    font-size:14px; font-weight:500; white-space:nowrap;
    transition: all .2s; color: var(--muted);
}
.tab:hover { background: rgba(255,255,255,.04); color: var(--text); }
.tab.active { background: rgba(102,126,234,.15); color: var(--blue); }
.tab-content { display:none; animation: fadeIn .3s ease; }
.tab-content.active { display:block; }
@keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }

/* ── Cards & Grid ───────────────────────────── */
.grid { display:grid; gap:16px; margin-bottom:20px; }
.grid-4 { grid-template-columns: repeat(auto-fit, minmax(240px,1fr)); }
.grid-3 { grid-template-columns: repeat(auto-fit, minmax(300px,1fr)); }
.grid-2 { grid-template-columns: repeat(auto-fit, minmax(400px,1fr)); }
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 22px;
}
.card-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; }
.card h3 { font-size:15px; font-weight:600; color:var(--muted); }
.stat-value {
    font-size: 32px; font-weight: 700; letter-spacing: -0.5px;
    margin: 6px 0 2px;
}
.stat-sub { font-size:12px; color:var(--muted); }
.green { color: var(--green); }
.red { color: var(--red); }
.blue { color: var(--blue); }

/* ── Buttons ────────────────────────────────── */
.btn {
    padding: 12px 24px; border:none; border-radius:10px;
    font-size:14px; font-weight:600; cursor:pointer;
    transition: all .2s; display:inline-flex; align-items:center; gap:6px;
}
.btn:hover { transform:translateY(-1px); filter:brightness(1.1); }
.btn:active { transform:translateY(0); }
.btn-green  { background: linear-gradient(135deg, #10b981, #059669); color:#fff; }
.btn-red    { background: linear-gradient(135deg, #ef4444, #dc2626); color:#fff; }
.btn-blue   { background: linear-gradient(135deg, var(--blue), #5a6fd6); color:#fff; }
.btn-amber  { background: linear-gradient(135deg, var(--amber), #d97706); color:#fff; }
.btn-outline{
    background: transparent; border:1px solid var(--border); color:var(--text);
}
.btn-group { display:flex; gap:8px; flex-wrap:wrap; }

/* ── Table ──────────────────────────────────── */
.table-wrap { overflow-x:auto; border-radius:var(--radius); border:1px solid var(--border); }
table { width:100%; border-collapse:collapse; background: var(--surface); }
th { padding:14px 16px; text-align:left; font-size:12px; font-weight:600; text-transform:uppercase;
     letter-spacing:.5px; color:var(--muted); background:var(--surface2); }
td { padding:12px 16px; font-size:13px; border-top:1px solid var(--border); }
tr:hover td { background: rgba(255,255,255,.02); }
.pill {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:11px; font-weight:600;
}
.pill-buy  { background:rgba(16,185,129,.15); color:var(--green); }
.pill-sell { background:rgba(239,68,68,.15);  color:var(--red); }

/* ── Form ───────────────────────────────────── */
label { display:block; font-size:13px; color:var(--muted); margin-bottom:6px; font-weight:500; }
input, select {
    width:100%; padding:11px 14px;
    background: var(--surface2); border:1px solid var(--border);
    border-radius:8px; color:var(--text); font-size:14px;
    margin-bottom:14px; outline:none; transition: border .2s;
}
input:focus, select:focus { border-color: var(--blue); }

/* ── Strategy cards ─────────────────────────── */
.strat-card {
    display:flex; justify-content:space-between; align-items:center;
    padding:18px; background:var(--surface2); border-radius:12px;
    margin-bottom:10px; border:1px solid var(--border);
}
.strat-card h4 { font-size:15px; margin-bottom:4px; }
.strat-meta { font-size:12px; color:var(--muted); display:flex; gap:16px; }

/* ── Positions mini-table ───────────────────── */
.pos-row {
    display:grid; grid-template-columns: 1fr 60px 80px 80px 80px;
    padding:10px 14px; font-size:13px;
    border-bottom:1px solid var(--border);
}
.pos-row.header { font-weight:600; color:var(--muted); font-size:11px; text-transform:uppercase; }

/* ── Responsive ─────────────────────────────── */
@media(max-width:768px){
    .container{padding:12px;}
    .header{padding:12px 16px;}
    .header h1{font-size:17px;}
    .grid-4{grid-template-columns:repeat(2,1fr);}
    .stat-value{font-size:24px;}
    .tab{padding:10px 14px;font-size:12px;}
}
</style>
</head>
<body>

<!-- ═════════ HEADER ═════════ -->
<div class="header">
    <h1>🚀 Professional Trading Platform</h1>
    <div class="header-right">
        <span class="badge" id="ibkrBadge">IBKR: ...</span>
        <span class="badge" id="botBadge">Bot: ...</span>
    </div>
</div>

<div class="container">

<!-- ═════════ TABS ═════════ -->
<div class="tabs">
    <div class="tab active"  onclick="switchTab('dashboard',this)">📊 Dashboard</div>
    <div class="tab" onclick="switchTab('control',this)">🎮 Control</div>
    <div class="tab" onclick="switchTab('positions',this)">📍 Positions</div>
    <div class="tab" onclick="switchTab('strategies',this)">🎯 Strategies</div>
    <div class="tab" onclick="switchTab('trades',this)">📝 Trades</div>
    <div class="tab" onclick="switchTab('taxes',this)">💶 Taxes (DE)</div>
    <div class="tab" onclick="switchTab('ml',this)">🧠 ML Stats</div>
    <div class="tab" onclick="switchTab('config',this)">⚙️ Config</div>
</div>

<!-- ═════════ DASHBOARD ═════════ -->
<div id="dashboard" class="tab-content active">
    <div class="grid grid-4">
        <div class="card">
            <h3>💰 Total Capital</h3>
            <div class="stat-value green" id="capital">€500.00</div>
            <div class="stat-sub" id="capitalReturn">+0.00%</div>
        </div>
        <div class="card">
            <h3>📊 Net P&L (After Tax)</h3>
            <div class="stat-value" id="netPnl">€0.00</div>
            <div class="stat-sub" id="grossPnl">Gross: €0.00</div>
        </div>
        <div class="card">
            <h3>💶 Tax Reserve (26.375%)</h3>
            <div class="stat-value red" id="taxReserve">€0.00</div>
            <div class="stat-sub">Finanzamt</div>
        </div>
        <div class="card">
            <h3>📈 Win Rate</h3>
            <div class="stat-value blue" id="winRate">0%</div>
            <div class="stat-sub" id="winLoss">0W / 0L</div>
        </div>
    </div>

    <div class="card" style="margin-bottom:20px">
        <div class="card-header"><h3>📈 Equity Curve</h3></div>
        <canvas id="equityChart" height="90"></canvas>
    </div>

    <div class="grid grid-4">
        <div class="card">
            <h3>📍 Open Positions</h3>
            <div class="stat-value" id="positions">0</div>
        </div>
        <div class="card">
            <h3>📊 Sharpe Ratio</h3>
            <div class="stat-value" id="sharpe">0.00</div>
        </div>
        <div class="card">
            <h3>📉 Max Drawdown</h3>
            <div class="stat-value red" id="maxDD">0%</div>
        </div>
        <div class="card">
            <h3>🎯 Total Trades</h3>
            <div class="stat-value" id="totalTrades">0</div>
        </div>
    </div>
</div>

<!-- ═════════ CONTROL ═════════ -->
<div id="control" class="tab-content">
    <div class="grid grid-2">
        <div class="card">
            <div class="card-header"><h3>🎮 Bot Control</h3></div>
            <div class="btn-group" style="margin-bottom:20px">
                <button class="btn btn-green" onclick="startBot()">▶️ Start</button>
                <button class="btn btn-amber" onclick="pauseBot()">⏸️ Pause</button>
                <button class="btn btn-blue"  onclick="resumeBot()">⏯️ Resume</button>
                <button class="btn btn-red"   onclick="stopBot()">⏹️ Stop</button>
            </div>
            <div id="systemInfo" style="color:var(--muted);font-size:13px;line-height:1.8;">Loading...</div>
        </div>
        <div class="card">
            <div class="card-header"><h3>ℹ️ System Info</h3></div>
            <div id="systemDetails" style="font-size:13px;color:var(--muted);line-height:2;">
                <p>Platform: Professional Trading Bot v2.0</p>
                <p>Broker: Interactive Brokers (via ib-insync)</p>
                <p>ML: Momentum Predictor + Q-Learning RL</p>
                <p>Strategies: 5 active (Mean Rev, Momentum, Breakout, RSI, BB)</p>
                <p>Risk: Kelly Criterion + Drawdown Protection</p>
                <p>Tax: German 26.375% (KapESt + Soli)</p>
            </div>
        </div>
    </div>
</div>

<!-- ═════════ POSITIONS ═════════ -->
<div id="positions" class="tab-content">
    <div class="card">
        <div class="card-header"><h3>📍 Open Positions</h3></div>
        <div id="positionsList">
            <p style="color:var(--muted);text-align:center;padding:30px">No open positions</p>
        </div>
    </div>
</div>

<!-- ═════════ STRATEGIES ═════════ -->
<div id="strategies" class="tab-content">
    <div class="card">
        <div class="card-header"><h3>🎯 Active Strategies</h3></div>
        <div id="strategiesList"><p style="color:var(--muted);text-align:center;padding:30px">Loading...</p></div>
    </div>
</div>

<!-- ═════════ TRADES ═════════ -->
<div id="trades" class="tab-content">
    <div class="card" style="padding:0">
        <div style="padding:18px 22px;border-bottom:1px solid var(--border)">
            <h3>📝 Trade History</h3>
        </div>
        <div class="table-wrap">
            <table>
                <thead><tr>
                    <th>Date</th><th>Symbol</th><th>Side</th><th>Qty</th>
                    <th>Entry</th><th>Exit</th><th>P&L</th><th>Tax</th><th>Strategy</th><th>Reason</th>
                </tr></thead>
                <tbody id="tradesTable">
                    <tr><td colspan="10" style="text-align:center;color:var(--muted);padding:30px">No closed trades yet — start the bot to begin</td></tr>
                </tbody>
            </table>
        </div>
    </div>
</div>

<!-- ═════════ TAXES ═════════ -->
<div id="taxes" class="tab-content">
    <div class="grid grid-3">
        <div class="card">
            <h3>💶 Total Tax Owed</h3>
            <div class="stat-value red" id="totalTax">€0.00</div>
            <div class="stat-sub">26.375% (KapESt 25% + Soli 5.5%)</div>
        </div>
        <div class="card">
            <h3>📊 Taxable Profit YTD</h3>
            <div class="stat-value green" id="taxableProfit">€0.00</div>
            <div class="stat-sub">After loss offsetting</div>
        </div>
        <div class="card">
            <h3>📉 Total Losses</h3>
            <div class="stat-value" id="totalLosses">€0.00</div>
            <div class="stat-sub">Deductible against gains</div>
        </div>
    </div>
    <div class="card" style="margin-top:16px">
        <div class="card-header"><h3>📄 Export for Finanzamt</h3></div>
        <p style="color:var(--muted);margin-bottom:16px;font-size:14px">
            Generate a CSV report with all trades, P&L, and calculated tax amounts for your German tax declaration (Anlage KAP).
        </p>
        <button class="btn btn-blue" onclick="exportTax()">📄 Download Tax Report CSV</button>
    </div>
</div>

<!-- ═════════ ML STATS ═════════ -->
<div id="ml" class="tab-content">
    <div class="grid grid-3">
        <div class="card">
            <h3>🧠 Market Predictor</h3>
            <div class="stat-value blue" id="mlTrained">Not trained</div>
            <div class="stat-sub" id="mlSamples">0 training samples</div>
        </div>
        <div class="card">
            <h3>🤖 RL Agent</h3>
            <div class="stat-value" id="rlStates">0 states</div>
            <div class="stat-sub" id="rlEpsilon">ε = 0.10</div>
        </div>
        <div class="card">
            <h3>⚙️ Parameter Optimizer</h3>
            <div class="stat-value" id="paramSets">0 sets</div>
            <div class="stat-sub">tested</div>
        </div>
    </div>
    <div class="card" style="margin-top:16px">
        <div class="card-header"><h3>How the ML Engine Works</h3></div>
        <div style="color:var(--muted);font-size:14px;line-height:1.8">
            <p><strong style="color:var(--text)">Market Predictor:</strong> Lightweight neural signal that analyzes price momentum, returns, volatility, and RSI to predict market direction (UP/DOWN/NEUTRAL).</p>
            <p style="margin-top:10px"><strong style="color:var(--text)">RL Agent:</strong> Q-learning reinforcement learning agent that learns optimal actions (BUY/SELL/HOLD) from trade outcomes, balancing exploration vs exploitation.</p>
            <p style="margin-top:10px"><strong style="color:var(--text)">Signal Enhancement:</strong> Combines strategy confidence (50%), ML prediction (30%), and RL confidence (20%) for final trade decisions.</p>
        </div>
    </div>
</div>

<!-- ═════════ CONFIG ═════════ -->
<div id="config" class="tab-content">
    <div class="grid grid-2">
        <div class="card">
            <div class="card-header"><h3>⚙️ Trading Parameters</h3></div>
            <label>Initial Capital (€)</label>
            <input type="number" id="cfgCapital" value="500" step="100">
            <label>Min Signal Confidence (%)</label>
            <input type="number" id="cfgConf" value="60" min="30" max="95">
            <label>Max Position Size (%)</label>
            <input type="number" id="cfgPos" value="10" min="1" max="25">
            <label>Max Daily Loss (%)</label>
            <input type="number" id="cfgDailyLoss" value="5" min="1" max="15">
            <button class="btn btn-blue" onclick="saveConfig()" style="margin-top:6px">💾 Save Configuration</button>
        </div>
        <div class="card">
            <div class="card-header"><h3>🔗 Broker Connection</h3></div>
            <label>IBKR Host</label>
            <input type="text" id="cfgHost" value="127.0.0.1">
            <label>IBKR Port (7497=Paper, 7496=Live)</label>
            <input type="number" id="cfgPort" value="7497">
            <label>Client ID</label>
            <input type="number" id="cfgClientId" value="1">
            <p style="color:var(--muted);font-size:13px;margin-top:10px;line-height:1.6">
                ⚠️ For live trading, install <strong>Trader Workstation (TWS)</strong> or IB Gateway on your machine, then update the host/port here. The bot auto-falls back to demo mode if IBKR is unreachable.
            </p>
        </div>
    </div>
</div>

</div><!-- /container -->

<!-- ═════════ JAVASCRIPT ═════════ -->
<script>
let chart = null;
let equityData = [];

function switchTab(name, el) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    el.classList.add('active');
    document.getElementById(name).classList.add('active');
}

// ── API helpers ──────────────────────────────────
async function api(url, opts) {
    try {
        const r = await fetch(url, opts);
        return await r.json();
    } catch(e) {
        console.error('API error:', url, e);
        return { success: false };
    }
}

// ── Dashboard Update ─────────────────────────────
async function update() {
    const d = await api('/api/bot/status');
    if (!d.success) return;
    const s = d.data;

    // Badges
    setBadge('botBadge', s.running ? 'Bot: ✅ Running' : 'Bot: ⏸️ Stopped', s.running);
    setBadge('ibkrBadge',
        s.simulation_mode ? 'IBKR: 🎮 Demo' : (s.connected ? 'IBKR: ✅ Live' : 'IBKR: ❌'),
        !s.simulation_mode && s.connected,
        s.simulation_mode ? 'warn' : undefined
    );

    // Dashboard stats
    const cap = s.current_capital || 500;
    const pnl = s.total_pnl || 0;
    const tax = pnl > 0 ? pnl * 0.26375 : 0;
    const net = pnl - tax;
    const retPct = s.total_return_pct || 0;

    setText('capital', '€' + cap.toFixed(2));
    setColor('capital', cap >= 500);
    setText('capitalReturn', (retPct >= 0 ? '+' : '') + retPct.toFixed(2) + '%');

    setText('netPnl', '€' + net.toFixed(2));
    setColor('netPnl', net >= 0);
    setText('grossPnl', 'Gross: €' + pnl.toFixed(2));

    setText('taxReserve', '€' + tax.toFixed(2));
    setText('winRate', (s.win_rate || 0).toFixed(1) + '%');
    setText('winLoss', (s.winning_trades||0) + 'W / ' + (s.losing_trades||0) + 'L');
    setText('positions', s.open_positions || 0);
    setText('sharpe', (s.sharpe_ratio || 0).toFixed(2));
    setText('maxDD', (s.max_historical_drawdown || 0).toFixed(1) + '%');
    setText('totalTrades', s.total_trades || 0);

    // System info
    document.getElementById('systemInfo').innerHTML = `
        <p>Status: ${s.running ? '✅ Trading Active' : s.paused ? '⏸️ Paused' : '⏹️ Stopped'}</p>
        <p>Mode: ${s.simulation_mode ? '🎮 Demo (simulated prices)' : '💰 Live Trading'}</p>
        <p>Uptime: ${s.uptime || '0s'}</p>
        <p>Trades: ${s.trade_count || 0} executed</p>
        <p>Open Positions: ${s.open_positions || 0}</p>
    `;

    // Positions
    const posList = s.open_positions_list || [];
    if (posList.length > 0) {
        document.getElementById('positionsList').innerHTML =
            '<div class="pos-row header"><span>Symbol</span><span>Side</span><span>Entry</span><span>Current</span><span>P&L</span></div>' +
            posList.map(p => `
                <div class="pos-row">
                    <span><strong>${p.symbol}</strong></span>
                    <span><span class="pill ${p.direction==='BUY'?'pill-buy':'pill-sell'}">${p.direction}</span></span>
                    <span>$${p.entry_price.toFixed(2)}</span>
                    <span>$${p.current_price.toFixed(2)}</span>
                    <span style="color:${p.unrealized_pnl>=0?'var(--green)':'var(--red)'}">€${p.unrealized_pnl.toFixed(2)}</span>
                </div>
            `).join('');
    } else {
        document.getElementById('positionsList').innerHTML =
            '<p style="color:var(--muted);text-align:center;padding:30px">No open positions</p>';
    }

    // Strategies
    const strats = s.strategies || [];
    if (strats.length > 0) {
        document.getElementById('strategiesList').innerHTML = strats.map(st => `
            <div class="strat-card">
                <div>
                    <h4>${st.name}</h4>
                    <div class="strat-meta">
                        <span>Win Rate: ${st.win_rate.toFixed(1)}%</span>
                        <span>Trades: ${st.trades}</span>
                        <span>Score: ${st.performance_score.toFixed(2)}</span>
                    </div>
                </div>
                <span class="pill ${st.enabled?'pill-buy':'pill-sell'}">${st.enabled?'Active':'Off'}</span>
            </div>
        `).join('');
    }

    // ML stats
    const ml = s.ml_stats || {};
    setText('mlTrained', ml.predictor_trained ? '✅ Trained' : '⏳ Collecting data');
    setColor('mlTrained', ml.predictor_trained);
    setText('mlSamples', ml.training_samples + ' training samples');
    setText('rlStates', ml.rl_states_learned + ' states');
    setText('rlEpsilon', 'ε = ' + (ml.rl_epsilon || 0.1).toFixed(3) + ' (exploration)');
    setText('paramSets', ml.parameter_sets_tested + ' sets');

    // Equity chart
    equityData.push(cap);
    if (equityData.length > 100) equityData.shift();
    updateChart();
}

function setBadge(id, text, isOn, override) {
    const el = document.getElementById(id);
    el.textContent = text;
    el.className = 'badge ' + (override === 'warn' ? 'badge-warn' : (isOn ? 'badge-on' : 'badge-off'));
}
function setText(id, val) { document.getElementById(id).textContent = val; }
function setColor(id, isPositive) {
    const el = document.getElementById(id);
    el.classList.remove('green','red');
    el.classList.add(isPositive ? 'green' : 'red');
}

// ── Chart ────────────────────────────────────────
function updateChart() {
    const ctx = document.getElementById('equityChart');
    if (!ctx) return;

    if (chart) {
        chart.data.labels = equityData.map((_,i) => i);
        chart.data.datasets[0].data = equityData;
        chart.update('none');
        return;
    }

    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: equityData.map((_,i) => i),
            datasets: [{
                data: equityData,
                borderColor: '#667eea',
                backgroundColor: 'rgba(102,126,234,.1)',
                fill: true, tension: 0.4, pointRadius: 0, borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: {
                    grid: { color: 'rgba(255,255,255,.05)' },
                    ticks: { color: '#8892a7', callback: v => '€' + v.toFixed(0) }
                }
            }
        }
    });
}

// ── Bot Controls ─────────────────────────────────
async function startBot() {
    const d = await api('/api/bot/start', {method:'POST'});
    alert(d.success ? '✅ Bot started!' : '❌ ' + (d.error || 'Failed'));
    update();
}
async function stopBot() {
    if (!confirm('Stop bot and close all positions?')) return;
    const d = await api('/api/bot/stop', {method:'POST'});
    alert(d.success ? '⏹️ Bot stopped' : '❌ Failed');
    update();
}
async function pauseBot() {
    await api('/api/bot/pause', {method:'POST'});
    update();
}
async function resumeBot() {
    await api('/api/bot/resume', {method:'POST'});
    update();
}

// ── Trades ───────────────────────────────────────
async function loadTrades() {
    const d = await api('/api/trades');
    if (!d.success || !d.data || d.data.length === 0) return;
    document.getElementById('tradesTable').innerHTML = d.data.map(t => `
        <tr>
            <td>${t.entry_time ? new Date(t.entry_time).toLocaleDateString() : '-'}</td>
            <td><strong>${t.symbol}</strong></td>
            <td><span class="pill ${t.direction==='BUY'?'pill-buy':'pill-sell'}">${t.direction}</span></td>
            <td>${(t.quantity||0).toFixed(4)}</td>
            <td>$${(t.entry_price||0).toFixed(2)}</td>
            <td>${t.exit_price ? '$'+t.exit_price.toFixed(2) : '—'}</td>
            <td style="color:${(t.pnl||0)>=0?'var(--green)':'var(--red)'}; font-weight:600">€${(t.pnl||0).toFixed(2)}</td>
            <td>€${t.pnl > 0 ? (t.pnl * 0.26375).toFixed(2) : '0.00'}</td>
            <td>${t.strategy}</td>
            <td>${t.exit_reason || '—'}</td>
        </tr>
    `).join('');
}

// ── Taxes ────────────────────────────────────────
async function loadTaxes() {
    const d = await api('/api/taxes');
    if (!d.success) return;
    setText('totalTax', '€' + (d.data.total_tax||0).toFixed(2));
    setText('taxableProfit', '€' + (d.data.taxable_profit||0).toFixed(2));
    setText('totalLosses', '€' + (d.data.total_loss||0).toFixed(2));
}
function exportTax() { window.location.href = '/api/taxes/export'; }

// ── Config ───────────────────────────────────────
async function saveConfig() {
    const cfg = {
        initial_capital: parseFloat(document.getElementById('cfgCapital').value),
        min_confidence: parseFloat(document.getElementById('cfgConf').value) / 100,
        max_position_size_pct: parseFloat(document.getElementById('cfgPos').value),
        max_daily_loss_pct: parseFloat(document.getElementById('cfgDailyLoss').value)
    };
    const d = await api('/api/config', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify(cfg)
    });
    alert(d.success ? '✅ Config saved!' : '❌ Error');
}

// ── Init ─────────────────────────────────────────
update();
setInterval(update, 3000);
setInterval(loadTrades, 10000);
setInterval(loadTaxes, 15000);
loadTrades();
loadTaxes();
</script>
</body>
</html>'''

# ═══════════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print("=" * 60)
    print("    🚀 PROFESSIONAL TRADING PLATFORM v2.0")
    print("=" * 60)
    print(f"    URL: http://0.0.0.0:{port}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
