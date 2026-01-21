"""Calibration model fitting functions."""

import numpy as np


def fit_linear_weights(
    predictions: np.ndarray,
    label_matrix: np.ndarray,
    label_names: list[str],
) -> tuple[dict[str, float], float]:
    """Fit linear weights to predict predictions from label matrix.

    Model: prediction = sum(weight_i * label_i) + bias
    Constrained to produce outputs in [0, 1] range.

    Args:
        predictions: Target predictions (0-1)
        label_matrix: Feature matrix [n_samples, n_labels]
        label_names: Names of labels

    Returns:
        Tuple of (weights dict, bias)
    """
    from sklearn.linear_model import Ridge

    # Use Ridge regression with small regularization for stability
    model = Ridge(alpha=0.01, fit_intercept=True)
    model.fit(label_matrix, predictions)

    weights = {name: float(coef) for name, coef in zip(label_names, model.coef_)}
    bias = float(model.intercept_)

    return weights, bias


def fit_log_linear_weights(
    predictions: np.ndarray,
    label_matrix: np.ndarray,
    label_names: list[str],
) -> tuple[dict[str, float], float]:
    """Fit log-linear weights to predict predictions from label matrix.

    Model: prediction = sigmoid(sum(weight_i * label_i) + bias)

    Args:
        predictions: Target predictions (0-1)
        label_matrix: Feature matrix [n_samples, n_labels]
        label_names: Names of labels

    Returns:
        Tuple of (weights dict, bias)
    """
    from scipy.optimize import minimize
    from scipy.special import expit as sigmoid

    def loss(params: np.ndarray) -> float:
        weights = params[:-1]
        bias = params[-1]
        logits = label_matrix @ weights + bias
        preds = sigmoid(logits)
        # MSE loss
        return float(np.mean((preds - predictions) ** 2))

    # Initialize with zeros
    n_labels = len(label_names)
    initial_params = np.zeros(n_labels + 1)

    result = minimize(loss, initial_params, method="L-BFGS-B")

    weights = {name: float(coef) for name, coef in zip(label_names, result.x[:-1])}
    bias = float(result.x[-1])

    return weights, bias
