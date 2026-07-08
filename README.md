# WorldCup2026

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC.svg)](.github/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Probabilistic forecasting and Monte Carlo simulation for the FIFA World Cup.
The project combines dynamic team strength, FIFA and ClubElo ratings, recent
form, squad composition, market value, and a Dixon-Coles-adjusted score model.

## Results

Each fold trains on every completed international match available before the
tournament. The final holdout contains all 128 matches from the 2018 and 2022
World Cups; model and hyperparameter selection use earlier tournaments.

| Metric | Selected model | Poisson baseline |
|---|---:|---:|
| Ranked Probability Score | **0.1997** | 0.2099 |
| Log loss | **0.9651** | 1.0086 |
| Multiclass Brier score | **0.5656** | 0.5903 |
| Expected calibration error | **0.0453** | 0.0887 |
| Accuracy | **57.03%** | 55.47% |
| One-vs-rest AUC | **0.6929** | 0.6572 |

For applications that require a single 1X2 pick, the separately validated
`top1_accuracy_blend` reaches 57.81% on the final holdout and 59.06% in the
five-tournament diagnostic. The probability model remains the selected model
for proper scoring rules and simulation. The accuracy gains are not
statistically significant by tournament (`p = 0.50` on the final holdout and
`p = 0.1875` across the five-fold diagnostic).

The broader walk-forward diagnostic covers 320 matches from the 2006, 2010,
2014, 2018, and 2022 World Cups:

| Metric | Probability model | Top-1 policy | Poisson |
|---|---:|---:|---:|
| Ranked Probability Score | **0.1940** | 0.1962 | 0.2008 |
| Log loss | **0.9528** | 0.9638 | 0.9796 |
| Multiclass Brier score | **0.5609** | 0.5660 | 0.5760 |
| Expected calibration error | **0.0355** | 0.0717 | 0.0741 |
| Accuracy | 56.56% | **59.06%** | 57.50% |
| One-vs-rest AUC | **0.6847** | 0.6791 | 0.6738 |

On the final holdout, the mean RPS improvement over Poisson is `0.01021`.
The tournament-clustered bootstrap 95% interval is
`[-0.00504, 0.02702]`; the one-sided tournament sign-flip test gives
`p = 0.25`. The corresponding five-tournament diagnostic gives `p = 0.1875`.

## Model

The selected `market_consensus_hybrid` model uses two
histogram-gradient-boosted Poisson heads to estimate home and away scoring
rates. Score distributions are converted into win/draw/loss probabilities,
adjusted with a Dixon-Coles low-score correction, and blended with 6% de-vigged
public bookmaker consensus when matching pre-game odds are available.

The optional top-1 policy blends these probabilities with the Poisson reference
for a more conservative class decision. Its 67.5% Poisson weight was selected
on the 2006, 2010 and 2014 World Cups.

Features include:

- pre-match dynamic Elo and lagged Elo trajectory;
- FIFA rank and points, with explicit missingness indicators;
- independent external Elo;
- recent and competitive form, opponent-adjusted strength, and schedule load;
- squad age, caps, goals, positional balance, and manager continuity;
- ClubElo aggregates and league diversity;
- timestamped pre-tournament market-value aggregates.

The simulator implements the 48-team format, group tie-breaks, the official
495-case Annex C mapping for third-placed teams, knockout extra time and
penalties, and coherent bracket propagation.

## Quick Start

```bash
python -m venv .venv
```

Activate the environment, then install the package:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[research,dev]"
```

Run tests and inspect the CLI:

```bash
python -m pytest
worldcup2026 --help
```

### Run the model

```bash
worldcup2026 run --simulations 100000
```

This trains and evaluates the candidate models, selects the best validated
configuration, and runs the official-format tournament simulation. Public
source data are downloaded and cached on first use.

## CUDA

CUDA acceleration is used automatically when supported. The same backends run
on CPU when CUDA is unavailable.

```bash
python -m pip install -e ".[research,gpu,dev]"
worldcup2026 run --simulations 100000
```

CUDA availability depends on the installed drivers and on the binary support
provided by XGBoost, LightGBM, and CatBoost.

## Repository Layout

```text
src/worldcup2026/   data, models, evaluation, benchmarking, and simulation
tests/              unit and data-contract tests
data/external/      competition inputs and optional-data schemas
data/raw/           downloaded cache, ignored by Git
artifacts/          curated report and generated results
```

Raw third-party data, model caches, rendered PDFs, and tuning grids are not
versioned.

Paths can be redirected without editing code:

```bash
export WORLDCUP2026_DATA_DIR=/mnt/datasets/worldcup2026
export WORLDCUP2026_ARTIFACT_DIR=/mnt/results/worldcup2026
```

Windows PowerShell uses `$env:WORLDCUP2026_DATA_DIR = "D:\datasets\worldcup2026"`.
All supported variables are listed in [.env.example](.env.example).

## Reproducibility

Validation folds are chronological, ratings are updated sequentially, and
external features use dated snapshots. Random seeds are fixed for stochastic
estimators and simulations. Exact results may vary slightly across platforms
and dependency builds.

## Curated Outputs

- [Audited public benchmark table](artifacts/research_sota_comparison_audited.csv)
- [Combined model metrics](artifacts/research_model_metrics_combined.csv)
- [100,000-run 2026 forecast](artifacts/research_forecast_2026_100k.csv)

## Data Sources

The pipeline downloads public data from their original hosts, including
[martj42/international_results](https://github.com/martj42/international_results),
[ClubElo](http://clubelo.com/), Wikipedia squad pages, and Kaggle-hosted FIFA
ranking and squad datasets. Match-odds consensus comes from
[Football-Data](https://www.football-data.co.uk/data.php). Each source retains
its own licence and terms; downloaded copies are excluded from this repository.

FIFA names and competition materials are used only for identification and
research. This project is independent and is not endorsed by FIFA.

## Licence

Source code is released under the [MIT License](LICENSE). Dataset licences are
not sublicensed by this project.
