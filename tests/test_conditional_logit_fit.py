import numpy as np
import pytest
from scipy.optimize import approx_fprime
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from choicekit import ConditionalLogit
from choicekit._data import _normalize_choice_fit_data


def _synthetic_long_format_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.array(
        [
            [2.0, 1.0],
            [0.5, 1.0],
            [1.5, 0.0],
            [0.0, 0.0],
            [0.3, 0.2],
            [2.0, 0.2],
            [0.1, 1.5],
            [1.0, 1.5],
        ],
        dtype=float,
    )
    y = np.array([1, 0, 1, 0, 0, 1, 0, 1], dtype=int)
    choice_set_id = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=int)
    return X, y, choice_set_id


def test_sklearn_params_and_clone() -> None:
    estimator = ConditionalLogit(max_iter=77, tol=1e-6)
    assert estimator.get_params() == {"max_iter": 77, "tol": 1e-6}
    estimator.set_params(max_iter=55)
    assert estimator.max_iter == 55
    cloned = clone(estimator)
    assert isinstance(cloned, ConditionalLogit)
    assert cloned.get_params() == estimator.get_params()


def test_fit_produces_expected_attributes_and_improves_likelihood() -> None:
    X, y, choice_set_id = _synthetic_long_format_data()
    model = ConditionalLogit(max_iter=300, tol=1e-10).fit(
        X,
        y,
        choice_set_id=choice_set_id,
    )

    assert model.coef_.shape == (X.shape[1],)
    assert np.isfinite(model.coef_).all()
    assert model.converged_ is True
    assert model.n_iter_ >= 1
    assert isinstance(model.message_, str)
    assert model.n_features_in_ == X.shape[1]
    assert model.n_choice_sets_ == len(np.unique(choice_set_id))
    assert model.log_likelihood_ == pytest.approx(-model.negative_log_likelihood_)

    nll_zero, _ = model._negative_log_likelihood_and_gradient(
        np.zeros(X.shape[1], dtype=float),
        _normalize_choice_fit_data(
            X,
            y,
            choice_set_id=choice_set_id,
            alternative_id=None,
            individual_id=None,
        ),
    )
    assert model.negative_log_likelihood_ < nll_zero
    assert np.linalg.norm(model.coef_) > 0.01


def test_fit_time_init_params_is_supported_and_validated() -> None:
    X, y, choice_set_id = _synthetic_long_format_data()
    model = ConditionalLogit()
    model.fit(X, y, choice_set_id=choice_set_id, init_params=np.array([0.1, -0.2]))
    with pytest.raises(ValueError, match="init_params"):
        model.fit(X, y, choice_set_id=choice_set_id, init_params=np.array([0.1]))


def test_non_convergence_emits_warning_and_sets_attributes() -> None:
    X, y, choice_set_id = _synthetic_long_format_data()
    with pytest.warns(ConvergenceWarning):
        model = ConditionalLogit(max_iter=1, tol=1e-12).fit(
            X,
            y,
            choice_set_id=choice_set_id,
        )
    assert model.converged_ is False
    assert model.coef_.shape == (X.shape[1],)
    assert model.optimization_result_ is not None


def test_analytic_gradient_matches_finite_difference() -> None:
    X, y, choice_set_id = _synthetic_long_format_data()
    model = ConditionalLogit()
    data = _normalize_choice_fit_data(
        X,
        y,
        choice_set_id=choice_set_id,
        alternative_id=None,
        individual_id=None,
    )
    beta = np.array([0.3, -0.4], dtype=float)

    def objective(params: np.ndarray) -> float:
        value, _ = model._negative_log_likelihood_and_gradient(params, data)
        return value

    _, analytic_grad = model._negative_log_likelihood_and_gradient(beta, data)
    finite_diff_grad = approx_fprime(beta, objective, epsilon=1e-8)
    assert np.allclose(analytic_grad, finite_diff_grad, atol=1e-6, rtol=1e-5)
