from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class _ChoiceData:
    X: np.ndarray
    y: np.ndarray
    X_grouped: np.ndarray
    y_grouped: np.ndarray
    group_starts: np.ndarray
    group_stops: np.ndarray
    n_choice_sets: int
    n_features: int


def _normalize_choice_fit_data(
    X,
    y,
    *,
    choice_set_id,
    alternative_id=None,
    individual_id=None,
) -> _ChoiceData:
    if sparse.issparse(X):
        raise ValueError("X must be dense long-format choice data.")

    X_array = _to_2d_float_array(X)
    n_rows, n_features = X_array.shape

    y_array = _to_1d_array(y, name="y", expected_length=n_rows)
    if not np.isin(y_array, (0, 1)).all():
        raise ValueError("y must contain only binary values 0/1.")
    y_array = y_array.astype(float, copy=False)

    choice_set_array = _to_1d_array(
        choice_set_id,
        name="choice_set_id",
        expected_length=n_rows,
    )
    _validate_no_missing(choice_set_array, name="choice_set_id")

    if alternative_id is not None:
        alternative_id_array = _to_1d_array(
            alternative_id,
            name="alternative_id",
            expected_length=n_rows,
        )
        _validate_no_missing(alternative_id_array, name="alternative_id")
    else:
        alternative_id_array = None

    if individual_id is not None:
        individual_id_array = _to_1d_array(
            individual_id,
            name="individual_id",
            expected_length=n_rows,
        )
        _validate_no_missing(individual_id_array, name="individual_id")
        _validate_choice_set_individual_mapping(choice_set_array, individual_id_array)

    codes, first_seen_values = _factorize_first_seen(choice_set_array)
    n_choice_sets = first_seen_values.shape[0]
    if n_choice_sets < 1:
        raise ValueError("At least one Choice Set is required.")

    order = np.argsort(codes, kind="mergesort")
    X_grouped = X_array[order]
    y_grouped = y_array[order]
    grouped_codes = codes[order]
    group_starts, group_stops = _group_boundaries(grouped_codes, n_choice_sets)

    for group_idx in range(n_choice_sets):
        start = int(group_starts[group_idx])
        stop = int(group_stops[group_idx])
        group_size = stop - start
        if group_size < 2:
            group_label = first_seen_values[group_idx]
            raise ValueError(
                f"Choice Set {group_label!r} must contain at least two "
                "Alternative Rows."
            )

        group_chosen = y_grouped[start:stop].sum()
        if group_chosen != 1.0:
            group_label = first_seen_values[group_idx]
            raise ValueError(
                f"Choice Set {group_label!r} must contain exactly one "
                "Chosen Alternative."
            )

        if alternative_id_array is not None:
            group_alternatives = alternative_id_array[order][start:stop]
            if np.unique(group_alternatives).size != group_alternatives.size:
                group_label = first_seen_values[group_idx]
                raise ValueError(
                    f"alternative_id must be unique within Choice Set {group_label!r}."
                )

    return _ChoiceData(
        X=X_array,
        y=y_array,
        X_grouped=X_grouped,
        y_grouped=y_grouped,
        group_starts=group_starts,
        group_stops=group_stops,
        n_choice_sets=n_choice_sets,
        n_features=n_features,
    )


def _to_2d_float_array(X) -> np.ndarray:
    if hasattr(X, "to_numpy"):
        array = np.asarray(X.to_numpy(), dtype=float)
    else:
        array = np.asarray(X, dtype=float)

    if array.ndim != 2:
        raise ValueError("X must be a two-dimensional array-like object.")
    if array.shape[0] == 0:
        raise ValueError("X must contain at least one Alternative Row.")
    if not np.isfinite(array).all():
        raise ValueError("X must contain only finite numeric values.")
    return array


def _to_1d_array(values, *, name: str, expected_length: int) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        array = np.asarray(values.to_numpy())
    else:
        array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.shape[0] != expected_length:
        raise ValueError(
            f"{name} must have the same number of rows as X ({expected_length})."
        )
    return array


def _validate_no_missing(values: np.ndarray, *, name: str) -> None:
    for idx, value in enumerate(values):
        if value is None:
            raise ValueError(f"{name} contains missing values at row {idx}.")
        try:
            if bool(np.isnan(value)):
                raise ValueError(f"{name} contains missing values at row {idx}.")
        except TypeError:
            continue


def _factorize_first_seen(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mapping: dict[object, int] = {}
    codes = np.empty(values.shape[0], dtype=np.int64)
    unique_values: list[object] = []
    next_code = 0
    for idx, value in enumerate(values):
        if value not in mapping:
            mapping[value] = next_code
            unique_values.append(value)
            next_code += 1
        codes[idx] = mapping[value]
    return codes, np.asarray(unique_values, dtype=object)


def _group_boundaries(
    codes: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray]:
    group_starts = np.empty(n_groups, dtype=np.int64)
    group_stops = np.empty(n_groups, dtype=np.int64)
    for group_idx in range(n_groups):
        members = np.flatnonzero(codes == group_idx)
        group_starts[group_idx] = members[0]
        group_stops[group_idx] = members[-1] + 1
    return group_starts, group_stops


def _validate_choice_set_individual_mapping(
    choice_set_id: np.ndarray,
    individual_id: np.ndarray,
) -> None:
    mapping: dict[object, object] = {}
    for idx, choice_set_value in enumerate(choice_set_id):
        individual_value = individual_id[idx]
        if choice_set_value not in mapping:
            mapping[choice_set_value] = individual_value
            continue
        if mapping[choice_set_value] != individual_value:
            raise ValueError(
                "Each choice_set_id must map to at most one individual_id."
            )
