"""Probability metrics, calibration, and ensemble optimization."""

from __future__ import annotations

import math
from itertools import product

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


TOP1_POISSON_WEIGHT = 0.675


def safe_probability(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=float)
    probability = np.clip(probability, 1e-8, 1.0)
    return probability / probability.sum(axis=1, keepdims=True)


def rps_vector(probability: np.ndarray, truth: np.ndarray) -> np.ndarray:
    one_hot = np.eye(3)[truth]
    predicted_cdf = np.cumsum(probability, axis=1)
    observed_cdf = np.cumsum(one_hot, axis=1)
    return np.sum((predicted_cdf - observed_cdf) ** 2, axis=1) / 2.0


def brier_vector(probability: np.ndarray, truth: np.ndarray) -> np.ndarray:
    one_hot = np.eye(3)[truth]
    return np.sum((probability - one_hot) ** 2, axis=1)


def expected_calibration_error(
    probability: np.ndarray,
    truth: np.ndarray,
    bins: int = 10,
) -> float:
    confidence = probability.max(axis=1)
    correct = (probability.argmax(axis=1) == truth).astype(float)
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence >= low) & (
            confidence < high if high < 1 else confidence <= high
        )
        if mask.any():
            value += float(
                mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
            )
    return value


def metric_summary(
    probability: np.ndarray,
    truth: np.ndarray,
) -> dict[str, float]:
    probability = safe_probability(probability)
    truth = np.asarray(truth, dtype=int)
    try:
        auc = roc_auc_score(
            truth,
            probability,
            multi_class="ovr",
            labels=[0, 1, 2],
        )
    except ValueError:
        auc = float("nan")
    return {
        "n": float(len(truth)),
        "rps": float(rps_vector(probability, truth).mean()),
        "log_loss": float(log_loss(truth, probability, labels=[0, 1, 2])),
        "brier": float(brier_vector(probability, truth).mean()),
        "ece": expected_calibration_error(probability, truth),
        "accuracy": float(accuracy_score(truth, probability.argmax(axis=1))),
        "auc_ovr": float(auc),
    }


def clustered_bootstrap_interval(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    iterations: int = 10_000,
    seed: int = 2026,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    if len(values) != len(clusters) or not len(unique):
        raise ValueError("Values and non-empty clusters must have equal length.")
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations)
    grouped = [values[clusters == cluster] for cluster in unique]
    for iteration in range(iterations):
        sampled_groups = rng.integers(0, len(grouped), len(grouped))
        samples = []
        for group_index in sampled_groups:
            group = grouped[group_index]
            samples.append(group[rng.integers(0, len(group), len(group))])
        estimates[iteration] = np.concatenate(samples).mean()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def clustered_sign_flip_pvalue(
    values: np.ndarray,
    clusters: np.ndarray,
) -> float:
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    unique = np.unique(clusters)
    if len(values) != len(clusters) or not len(unique):
        raise ValueError("Values and non-empty clusters must have equal length.")
    cluster_sums = np.array([values[clusters == cluster].sum() for cluster in unique])
    observed = float(cluster_sums.sum())
    null_statistics = np.array(
        [
            float(np.dot(signs, cluster_sums))
            for signs in product((-1.0, 1.0), repeat=len(unique))
        ]
    )
    return float(np.mean(null_statistics >= observed - 1e-15))


def top1_decision_probabilities(
    probabilistic: np.ndarray,
    poisson: np.ndarray,
    poisson_weight: float = TOP1_POISSON_WEIGHT,
) -> np.ndarray:
    """Blend probabilities for the separately validated top-1 decision policy."""
    return safe_probability(
        (1.0 - poisson_weight) * probabilistic + poisson_weight * poisson
    )


def tune_temperature(probability: np.ndarray, truth: np.ndarray) -> float:
    probability = safe_probability(probability)

    def objective(log_temperature: np.ndarray) -> float:
        temperature = math.exp(float(log_temperature[0]))
        calibrated = safe_probability(probability ** (1.0 / temperature))
        return log_loss(truth, calibrated, labels=[0, 1, 2])

    result = minimize(
        objective,
        x0=np.array([0.0]),
        bounds=[(math.log(0.35), math.log(3.5))],
    )
    return math.exp(float(result.x[0]))


def apply_temperature(
    probability: np.ndarray,
    temperature: float,
) -> np.ndarray:
    return safe_probability(safe_probability(probability) ** (1.0 / temperature))


def fit_beta_calibrator(
    probability: np.ndarray,
    truth: np.ndarray,
) -> list[LogisticRegression]:
    probability = safe_probability(probability)
    models = []
    for label in range(3):
        label_probability = probability[:, label]
        features = np.column_stack(
            [
                np.log(np.clip(label_probability, 1e-8, 1.0)),
                np.log(np.clip(1.0 - label_probability, 1e-8, 1.0)),
            ]
        )
        target = (truth == label).astype(int)
        models.append(LogisticRegression(C=1.0, max_iter=1000).fit(features, target))
    return models


def apply_beta_calibrator(
    models: list[LogisticRegression],
    probability: np.ndarray,
) -> np.ndarray:
    probability = safe_probability(probability)
    calibrated = []
    for label, model in enumerate(models):
        label_probability = probability[:, label]
        features = np.column_stack(
            [
                np.log(np.clip(label_probability, 1e-8, 1.0)),
                np.log(np.clip(1.0 - label_probability, 1e-8, 1.0)),
            ]
        )
        calibrated.append(model.predict_proba(features)[:, 1])
    return safe_probability(np.column_stack(calibrated))


def fit_platt_calibrator(
    probability: np.ndarray,
    truth: np.ndarray,
) -> LogisticRegression:
    features = np.log(safe_probability(probability))
    model = LogisticRegression(C=1.0, max_iter=1000)
    return model.fit(features, truth)


def apply_platt_calibrator(
    model: LogisticRegression,
    probability: np.ndarray,
) -> np.ndarray:
    return safe_probability(model.predict_proba(np.log(safe_probability(probability))))


def optimize_ensemble(
    probabilities: list[np.ndarray],
    truth: np.ndarray,
) -> np.ndarray:
    stacked = np.stack(probabilities, axis=0)

    def objective(raw: np.ndarray) -> float:
        weights = np.exp(raw - raw.max())
        weights /= weights.sum()
        blended = np.einsum("m,mnc->nc", weights, stacked)
        return metric_summary(blended, truth)["rps"]

    result = minimize(objective, np.zeros(len(probabilities)), method="BFGS")
    weights = np.exp(result.x - result.x.max())
    return weights / weights.sum()
