"""Lab 6: bootstrap confidence intervals."""

import numpy as np


def bootstrap_ci(values, *, n_boot=2000, seed=42, alpha=0.05):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        raise ValueError("values must not be empty")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    rng = np.random.default_rng(seed)

    point = float(np.mean(values))

    samples = rng.choice(
        values,
        size=(n_boot, values.size),
        replace=True,
    )

    boot_means = np.mean(samples, axis=1)

    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))

    return point, lo, hi


def paired_bootstrap_diff(a, b, *, n_boot=2000, seed=42, alpha=0.05):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.size == 0 or b.size == 0:
        raise ValueError("a and b must not be empty")
    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    rng = np.random.default_rng(seed)

    differences = a - b
    delta = float(np.mean(differences))

    indices = rng.integers(
        0,
        differences.size,
        size=(n_boot, differences.size),
    )

    boot_diffs = np.mean(
        differences[indices],
        axis=1,
    )

    lo = float(np.quantile(boot_diffs, alpha / 2))
    hi = float(np.quantile(boot_diffs, 1 - alpha / 2))

    return delta, lo, hi