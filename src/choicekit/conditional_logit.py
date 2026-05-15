from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning

from ._data import _ChoiceData, _normalize_choice_fit_data


class ConditionalLogit(BaseEstimator):
    def __init__(self, *, max_iter: int = 1000, tol: float = 1e-8) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        X,
        y,
        *,
        choice_set_id,
        alternative_id=None,
        individual_id=None,
        init_params=None,
    ) -> ConditionalLogit:
        self._validate_hyperparameters()
        data = _normalize_choice_fit_data(
            X,
            y,
            choice_set_id=choice_set_id,
            alternative_id=alternative_id,
            individual_id=individual_id,
        )
        beta0 = self._validate_init_params(init_params, n_features=data.n_features)

        def objective(beta: np.ndarray) -> float:
            value, _ = self._negative_log_likelihood_and_gradient(beta, data)
            return value

        def gradient(beta: np.ndarray) -> np.ndarray:
            _, grad = self._negative_log_likelihood_and_gradient(beta, data)
            return grad

        result = minimize(
            objective,
            beta0,
            method="L-BFGS-B",
            jac=gradient,
            options={"maxiter": int(self.max_iter), "ftol": float(self.tol)},
        )

        final_nll, _ = self._negative_log_likelihood_and_gradient(
            np.asarray(result.x, dtype=float),
            data,
        )
        self.coef_ = np.asarray(result.x, dtype=float)
        self.negative_log_likelihood_ = float(final_nll)
        self.log_likelihood_ = -self.negative_log_likelihood_
        self.converged_ = bool(result.success)
        self.n_iter_ = int(getattr(result, "nit", 0))
        self.message_ = str(result.message)
        self.optimization_result_ = result
        self.n_features_in_ = int(data.n_features)
        self.n_choice_sets_ = int(data.n_choice_sets)

        if not self.converged_:
            warnings.warn(self.message_, ConvergenceWarning, stacklevel=2)

        return self

    def _validate_hyperparameters(self) -> None:
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol must be a positive finite float.")

    def _validate_init_params(
        self,
        init_params,
        *,
        n_features: int,
    ) -> np.ndarray:
        if init_params is None:
            return np.zeros(n_features, dtype=float)
        params = np.asarray(init_params, dtype=float)
        if params.ndim != 1:
            raise ValueError(
                "init_params must be one-dimensional with shape (n_features,)."
            )
        if params.shape[0] != n_features:
            raise ValueError(
                f"init_params must have shape ({n_features},), got {params.shape}."
            )
        if not np.isfinite(params).all():
            raise ValueError("init_params must contain only finite values.")
        return params

    def _negative_log_likelihood_and_gradient(
        self,
        coef: np.ndarray,
        data: _ChoiceData,
    ) -> tuple[float, np.ndarray]:
        utilities = data.X_grouped @ coef
        gradient = np.zeros(data.n_features, dtype=float)
        nll = 0.0

        for start, stop in zip(data.group_starts, data.group_stops, strict=True):
            utilities_group = utilities[start:stop]
            y_group = data.y_grouped[start:stop]
            X_group = data.X_grouped[start:stop]

            max_utility = float(np.max(utilities_group))
            shifted = utilities_group - max_utility
            exp_shifted = np.exp(shifted)
            denominator = float(exp_shifted.sum())
            probabilities = exp_shifted / denominator

            nll += -float(np.dot(y_group, utilities_group)) + max_utility + np.log(
                denominator
            )
            gradient += X_group.T @ (probabilities - y_group)

        return float(nll), gradient
