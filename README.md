# Sporty Hero — Crash Game RNG Study

Automated data collection and statistical analysis system for SportyBet's **Sporty Hero** crash/multiplier game.

## Goal

Empirically determine whether the crash multiplier distribution exhibits any detectable statistical structure by:
1. Collecting thousands of crash multiplier values
2. Analyzing the distribution (exponential? geometric? custom?)
3. Detecting patterns, streaks, or exploitable sequences
4. Backtesting cashout strategies (fixed multiplier, adaptive, streak-based)

## Architecture

```
Browser (Chromium) ──▸ Playwright Bot ──▸ SQLite DB ──▸ Analysis Pipeline ──▸ Reports
```

## Components

| File | Purpose |
|------|---------|
| `src/bot.py` | Playwright-based crash value scraper |
| `src/db.py` | SQLite storage for crash history |
| `src/analyze.py` | Statistical analysis + pattern detection |
| `src/strategies.py` | Cashout strategy backtesting |

## Quick Start

```bash
# Install dependencies
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
python -m playwright install chromium

# Collect data (observe mode — scrapes crash values)
python -m src bot --rounds 1000

# Analyze collected data
python -m src analyze
```

## Data Collected Per Round

- **Crash multiplier** (e.g., 1.23x, 5.67x, 150.00x)
- **Round ID / timestamp**
- **Recent crash history** (displayed on screen)

## Key Questions

1. What is the empirical distribution of crash values?
2. Are there streaks (many low crashes in a row, then a high one)?
3. Is there autocorrelation between consecutive crashes?
4. Does the crash history shown on screen influence the next crash?
5. What is the optimal fixed cashout multiplier for EV?
6. Can adaptive strategies (e.g., bet more after N low crashes) beat the house edge?

## Tech Stack

- Python 3.12 + Playwright (browser automation)
- SQLite (data storage)
- pandas, scipy, numpy (analysis)
- matplotlib, seaborn (visualization)

## License

MIT