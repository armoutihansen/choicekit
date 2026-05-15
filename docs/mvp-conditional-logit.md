# MVP: Conditional Logit

## Product Promise

choicekit's MVP estimates conditional logit models on Long-Format Choice Data
and provides prediction-first, Choice Set-aware workflows that follow the
scikit-learn estimator protocol where compatible.

The MVP is not a full econometric inference package. It should prove that
choicekit can make long-format random utility modeling feel natural in
scikit-learn-shaped Python code.

## Public API

```python
from choicekit import ConditionalLogit
from choicekit.metrics import choice_accuracy_score, choice_log_loss
from choicekit.model_selection import choice_train_test_split
```

MVP module layout:

```text
src/choicekit/
  __init__.py
  _data.py
  conditional_logit.py
  metrics.py
  model_selection.py
```

### Estimator

```python
ConditionalLogit(
    max_iter=1000,
    tol=1e-8,
)
```

`ConditionalLogit` should subclass or behave like `sklearn.base.BaseEstimator`.
It should not subclass `ClassifierMixin` in the MVP because Alternatives are not
global classifier classes and `predict_proba` deliberately returns row-aligned
probabilities.

The MVP should have hard runtime dependencies on NumPy, SciPy, and
scikit-learn. Pandas support should be provided by accepting pandas objects when
users pass them, without making pandas a required runtime dependency.

Methods:

```python
fit(
    X,
    y,
    *,
    choice_set_id,
    alternative_id=None,
    individual_id=None,
    init_params=None,
)
decision_function(X, *, choice_set_id)
predict_proba(X, *, choice_set_id, alternative_id=None)
predict(X, *, choice_set_id, alternative_id)
score(X, y, *, choice_set_id, alternative_id=None)
log_loss(X, y, *, choice_set_id, alternative_id=None)
negative_log_likelihood(X, y, *, choice_set_id, alternative_id=None)
```

Learned attributes:

```python
coef_
log_likelihood_
negative_log_likelihood_
converged_
n_iter_
message_
optimization_result_
n_choice_sets_
n_features_in_
feature_names_in_  # only when fit with named pandas DataFrame columns
```

Do not expose `classes_`, `cov_type`, `standard_errors_`, `covariance_`,
`score_by_choice_set_`, `regularization`, `alpha`, `random_state`, or
`fit_intercept` in the MVP.

## Data Contract

Long-Format Choice Data is native:

- rows are Alternative Rows;
- Choice Sets are identified by `choice_set_id`;
- `y` is a row-level binary or boolean indicator;
- each Choice Set must contain at least two Alternative Rows;
- each Choice Set must have exactly one Chosen Alternative.

`choice_set_id`, `alternative_id`, and `individual_id` are metadata, not feature
columns. Users should pass only numeric model features in `X`.

`X` may be a dense pandas DataFrame, dense NumPy array, or compatible dense
array-like object. Sparse matrices are out of scope for the MVP and should raise
a clear error.
`y` and metadata may be pandas Series, NumPy arrays, lists, or categorical-like
arrays. Outputs are NumPy arrays by default.

If `X` is a DataFrame, preserve feature names in `feature_names_in_`. Prediction
with a DataFrame must use the same columns in the same order. Prediction with a
NumPy array validates only `n_features_in_`.

Missing values and non-finite values are rejected in `X`, `y`, and required
metadata. `X` must be numeric; users handle categorical encoding upstream.

`fit` should reject any feature with no within-Choice Set variation across all
Choice Sets. Such features, including global constants and Choice Set-level
variables repeated across all Alternative Rows in each Choice Set, are not
identified in conditional logit.

## Metadata Rules

`choice_set_id` is required for every estimator method.

`alternative_id` is:

- optional for `fit`;
- optional for `predict_proba`;
- required for `predict`;
- optional for `score`.

When provided, `alternative_id` must be unique within each Choice Set. Global
uniqueness is not required. Unseen Alternatives are allowed at prediction time
because the MVP learns only feature coefficients.

`individual_id` is accepted by `fit`, validated for length and missingness when
provided, and otherwise unused in the MVP.

`choice_set_id` must identify Choice Sets globally within each method call.
Grouping is by `choice_set_id` alone, not by `(individual_id, choice_set_id)`.
If `individual_id` is provided, each `choice_set_id` must be associated with at
most one `individual_id`; otherwise raise a clear error. Users with
per-individual occasion numbers should construct composite Choice Set IDs
upstream.

## Prediction Semantics

`decision_function` returns deterministic utility values:

```python
utilities.shape == (n_alternative_rows,)
```

It requires `choice_set_id` so prediction-like methods remain explicit about
choice metadata, but it only validates metadata length and missingness. It does
not require each Choice Set to contain at least two Alternative Rows because the
utility index is defined per Alternative Row.

`predict_proba` returns one probability per Alternative Row:

```python
proba.shape == (n_alternative_rows,)
```

Probabilities are aligned exactly to input row order and normalized within each
Choice Set. This is a deliberate deviation from scikit-learn's usual classifier
shape.

`predict` returns one predicted `alternative_id` per Choice Set:

```python
pred.shape == (n_choice_sets,)
```

Prediction order is first-seen Choice Set order in the input. Within a Choice
Set, ties are broken by first row order.

Singleton Choice Sets are rejected in `fit`, `predict_proba`, `predict`, `score`,
`log_loss`, and `negative_log_likelihood`.

## Scoring and Metrics

`score` returns Choice Set-level accuracy:

- compute row-aligned probabilities;
- within each Choice Set, compare the max-probability row to the observed Chosen
  Alternative row from `y`;
- return the fraction of Choice Sets predicted correctly.

`alternative_id` is optional for `score`. When provided, it should be validated
like other metadata, but row positions are sufficient to compute accuracy.

`negative_log_likelihood` returns total negative log likelihood across Choice
Sets.

`log_loss` returns average negative log likelihood per Choice Set.

Standalone metrics:

```python
choice_accuracy_score(y_true, y_score, *, choice_set_id, alternative_id=None)
choice_log_loss(y_true, y_proba, *, choice_set_id, eps=1e-15)
```

`choice_accuracy_score` accepts row-aligned scores where larger is better.
`choice_log_loss` accepts row-aligned probabilities, validates finite
nonnegative probabilities, validates normalization within Choice Sets within a
tolerance, clips selected probabilities to `[eps, 1]`, and returns average
negative log likelihood per Choice Set.

## Model Selection

Provide an index-returning split helper:

```python
train_idx, test_idx = choice_train_test_split(
    choice_set_id,
    test_size=0.2,
    train_size=None,
    random_state=None,
)
```

The helper returns positional integer indices and keeps all Alternative Rows from
the same Choice Set together. It should delegate to `GroupShuffleSplit` and avoid
duplicating full estimator validation.

The helper should not support stratification in the MVP.
Returned `train_idx` and `test_idx` should be sorted ascending so each split
preserves the original row order.

Documentation should also show direct manual use of `GroupShuffleSplit` and
`GroupKFold`. Direct `GridSearchCV` and `cross_val_score` support are out of
scope for the MVP.

## Internal Representation

Use a private `_ChoiceData` or equivalent helper in `_data.py` to normalize arrays and
metadata. It should preserve public row order while allowing stable private
grouped ordering for efficient grouped likelihood and softmax calculations.

Choice Set order is first-seen order, not lexicographic order.

Public outputs must always be restored to input row order where row-aligned, and
to first-seen Choice Set order where Choice Set-aligned.

## Optimization

Estimate by maximum likelihood with `scipy.optimize.minimize` using
`method="L-BFGS-B"`.

Required implementation details:

- stable log-sum-exp likelihood calculation;
- analytic gradient of the negative log likelihood;
- deterministic zero initialization by default;
- optional fit-time `init_params` with shape `(n_features,)`;
- no Hessian or standard errors in MVP.

On optimizer non-success:

- complete `fit`;
- set final coefficients and optimization attributes;
- set `converged_ = False`;
- set `message_`;
- emit `sklearn.exceptions.ConvergenceWarning`;
- allow prediction methods to run.

## Validation Error Style

Use plain `ValueError`, `sklearn.exceptions.NotFittedError`, and
`sklearn.exceptions.ConvergenceWarning`.

Error messages should use choicekit domain terms and include the first offending
Choice Set or Alternative when feasible.

## Required Documentation

The MVP is not complete without examples showing:

1. basic fit, probability prediction, prediction, and scoring on long-format
   data;
2. Choice Set-aware train/test splitting with `choice_train_test_split`;
3. manual grouped cross-validation with `GroupKFold`.

Documentation must explicitly warn users not to use row-wise
`train_test_split`, not to include metadata columns in `X`, and not to expect
2D classifier-style `predict_proba`.

## Out of Scope

The MVP excludes:

- mixed logit;
- latent-class logit;
- WTP-space models;
- alternative-specific constants;
- availability masks;
- regularization;
- sample weights;
- standard errors;
- robust or clustered covariance;
- summary tables;
- formula API;
- automatic categorical encoding;
- metadata routing;
- direct `GridSearchCV` and `cross_val_score` support;
- GPU acceleration;
- public `ChoiceDataset`.

## Definition of Done

The MVP is done when:

- `ConditionalLogit` fits known synthetic conditional-logit data;
- analytic gradient is tested against finite differences;
- `predict_proba` is row-aligned and sums to one within Choice Sets;
- shuffled row-order tests prove output alignment is preserved;
- validation errors cover malformed Choice Sets and metadata;
- pandas and NumPy inputs are both tested;
- sklearn `clone`, `get_params`, and `set_params` work;
- grouped train/test and grouped CV examples are documented;
- local checks pass with `ruff`, `pytest`, and package build.
