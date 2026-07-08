# WorldCup2026

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC.svg)](.github/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Reproducible pre-match football forecasting and FIFA World Cup simulation.

## Results

Models are selected by pooled Ranked Probability Score (RPS) on the 2006,
2010, and 2014 World Cups. The selected model is then evaluated retrospectively
on all 128 matches from 2018 and 2022. Targets are the result after 90 minutes;
extra-time goals are reconstructed from event-level data and excluded.

| Metric | Selected model | Poisson |
|---|---:|---:|
| RPS | **0.20553** | 0.21535 |
| Log loss | **0.99490** | 1.03724 |
| Multiclass Brier | **0.58533** | 0.60852 |
| Expected calibration error | 0.04716 | **0.07874** |
| Accuracy | **56.25%** | 55.47% |
| One-vs-rest AUC | **0.68344** | 0.65679 |

The selected `squad_hist_gradient_dc_prior` model improves mean RPS over Poisson
by `0.00982`. The tournament-clustered bootstrap 95% interval is
`[-0.01059, 0.01126]`; the tournament sign-flip test gives `p = 0.50`.
The difference is not statistically significant.

The closing-odds `market_consensus_hybrid` records a lower retrospective RPS
of `0.20831`, but it was not selected by the earlier development tournaments.
It is reported as an observed retrospective result, not promoted as the
deployed model.

The 2018/2022 editions were inspected in earlier iterations of this repository,
so this is not a pristine confirmatory test set. It is deliberately labelled
retrospective; prospective claims require a forecast registry entry published
before the corresponding matches.

Published point estimates whose targets or information sets cannot be matched
are marked non-comparable in
[`artifacts/research_sota_comparison_audited.csv`](artifacts/research_sota_comparison_audited.csv).
This project does not currently support a state-of-the-art claim.

## Method

The selected model has separate histogram-gradient-boosted Poisson heads for
each team's goals. It uses information available before each match:

- dynamic Elo and independent rating snapshots;
- FIFA rank and points;
- opponent-adjusted recent and competitive form;
- rolling xG, shots, possession, passing, corners, discipline, and availability
  counts from completed earlier matches;
- neutral-venue and signed host advantage.

Calibration choice is nested by tournament across temperature, Platt, beta,
isotonic, and no calibration. Candidate models include Elo logistic, Poisson,
Dixon-Coles, boosted goal models, boosted multiclass models, team-effect
Poisson, SDR, squad models, and a market consensus model.

The 2026 pre-tournament forecast is a separate estimand. Its model is selected
only among structural candidates and excludes closing odds and sources
published after the `2026-06-10` cutoff.

The simulator implements the 48-team format, FIFA group tie-breaks, Annex C,
the knockout bracket, 30-minute extra time, penalties, and signed host
advantage. The forecast contains binomial Monte Carlo intervals from 100,000
simulations; these intervals do not include model-parameter uncertainty.

## Install

Using the locked environment:

```bash
uv sync --all-extras
```

Or with pip:

```bash
python -m pip install -e ".[research,gpu,dev]"
```

Run verification and the experiment:

```bash
worldcup2026 verify-data
python -m pytest
worldcup2026 run --simulations 100000
```

CUDA is used automatically by supported XGBoost, LightGBM, and CatBoost
installations. CPU execution remains supported.

## Reproducibility

Downloaded data are excluded from Git. Their expected SHA-256 hashes and source
revisions are tracked in [`data/source_manifest.json`](data/source_manifest.json).
`uv.lock` fixes the Python dependency graph.

Forecasts can be registered with their model, cutoff, source-manifest hash, and
file hash:

```bash
worldcup2026 register-forecast artifacts/forecast.csv \
  --model MODEL --cutoff YYYY-MM-DD --status prospective
```

The included 2026 forecast is explicitly registered as retrospective because
it was generated after the tournament began. A registry entry is an audit
record; publication of its Git commit provides the external timestamp.

## Layout

```text
src/worldcup2026/  data pipeline, models, evaluation, and simulation
tests/             behavioral and temporal-data tests
data/external/     small versioned competition inputs
data/raw/          downloaded cache, ignored by Git
artifacts/         curated metrics, forecast, and registry
```

The model uses public data from
[martj42/international_results](https://github.com/martj42/international_results),
[ClubElo](http://clubelo.com/), Football-Data, Wikipedia, Hugging Face, and
Kaggle-hosted datasets. Third-party data retain their original licences.

## License

Code is released under the [MIT License](LICENSE). This project is independent
and is not endorsed by FIFA.
