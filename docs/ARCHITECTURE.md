# Architecture & System Design

## Core Flow
The WorldCup2026 repository operates as a robust, reproducible data pipeline. The framework extracts raw empirical data, standardizes it, and trains gradient-boosted ensembles to reconstruct and predict outcomes.

1. **Ingestion Layer (`pipeline.py`)**: Responsible for historical data parsing. Normalizes teams, evaluates venue advantages, and calculates complex multi-year features such as the *World Football Elo Ratings*.
2. **Feature Engineering (`match_stats.py`)**: Embeds tactical insights by extracting Expected Goals (xG), Possession, and Passing ratios using an *Exponential Moving Average (EMA)*.
3. **Simulation Engine (`goals_simulation.py`)**: Uses trained estimators (XGBoost, LightGBM, HistGradientBoosting) to simulate Monte Carlo paths for knockout scenarios.
4. **Evaluation Benchmark (`research_benchmark.py`)**: Executes strict walk-forward cross-validation testing to ensure models are validated retro-spectively without temporal leakage.

## Publication Boundaries
To maintain a high-quality open-source portfolio, the following components are explicitly kept outside the committed Git tree:
- **Raw Kaggle Archives**: `data/raw/` contains massive raw zips which are dynamically fetched by the `cached_download()` utility.
- **Model Checkpoints**: Only metric outputs (`artifacts/*.csv`) are committed.
- **Environment Variables**: Managed locally.
