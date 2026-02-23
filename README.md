# Baseball Analysis

Sabermetric analysis tools using MLB Statcast data.

## Scripts

### `predictivemodel.py`
Identifies "unlucky" hitters each season — players whose wOBA significantly underperformed their xWOBA — and backtests whether they bounced back the following year. Uses linear regression to estimate the trend going forward.

**Data required:** `stats.csv` (Statcast hitting data with `woba`, `xwoba`, `pa`, `year`, `player_id`, `batting_avg`)

### `stuff_vs_sequencing.py`
Analyzes whether elite pitchers rely on **raw stuff** or **pitch sequencing** to generate whiffs. For a given pitcher, it identifies their best pitch (by whiff rate) and tests whether it's more effective when set up by their most common other pitch.

**Pitchers analyzed:** Yoshinobu Yamamoto, Tarik Skubal, Paul Skenes

**Data required:** `yamamoto.csv`, `skubal.csv`, `skenes.csv` (Statcast pitch-by-pitch data from Baseball Savant)

## Setup

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Data
CSV files are not tracked in this repo. Download pitch-by-pitch data from [Baseball Savant](https://baseballsavant.mlb.com).
