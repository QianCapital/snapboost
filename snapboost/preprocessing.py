"""Optional preprocessing helpers for heterogeneous tabular data."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _as_column_sequence(categorical_features: object) -> tuple[Any, ...]:
    """Normalize a column selector to a sequence of names or positions.

    A string or integer is one column, matching scikit-learn column selectors.
    ``None`` means no categorical columns.
    """
    if categorical_features is None:
        return ()
    if isinstance(categorical_features, (bool, np.bool_)):
        raise ValueError(
            "categorical_features must be a column name, index, or a sequence "
            "of columns."
        )
    if isinstance(categorical_features, (str, bytes)):
        return (categorical_features,)
    if isinstance(categorical_features, Integral):
        return (int(categorical_features),)
    if isinstance(categorical_features, (Sequence, np.ndarray)):
        return tuple(categorical_features)
    raise ValueError(
        "categorical_features must be a column name, index, or a sequence "
        "of columns."
    )


def make_tabular_preprocessor(
    categorical_features: object = (),
    *,
    scale_numeric: bool = False,
    add_missing_indicators: bool = True,
) -> ColumnTransformer:
    """Build a dense, leakage-safe preprocessor for use before SnapBoost.

    Numeric columns are median-imputed, optionally augmented with missingness
    indicators and standardized. Categorical columns are most-frequent-imputed
    and one-hot encoded with unknown-category handling. The returned transformer
    belongs in a normal scikit-learn ``Pipeline`` and does not alter SnapBoost's
    internal training or model format.

    ``categorical_features`` may be a single column name or index, or a
    sequence of names/indices. A string is treated as one column name, not as
    a sequence of characters.
    """
    categorical_features = _as_column_sequence(categorical_features)
    numeric_steps = [
        (
            "impute",
            SimpleImputer(
                strategy="median",
                add_indicator=add_missing_indicators,
            ),
        )
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    encoder_options: dict[str, Any] = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        encoder_options["sparse_output"] = False
    else:
        encoder_options["sparse"] = False
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        (
            "encode",
            OneHotEncoder(**encoder_options),
        ),
    ])
    transformers = []
    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, list(categorical_features))
        )
    return ColumnTransformer(
        transformers=transformers,
        remainder=Pipeline(numeric_steps),
        sparse_threshold=0.0,
    )
