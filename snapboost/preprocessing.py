"""Optional preprocessing helpers for heterogeneous tabular data."""

import inspect

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_tabular_preprocessor(
    categorical_features=(),
    *,
    scale_numeric=False,
    add_missing_indicators=True,
):
    """Build a dense, leakage-safe preprocessor for use before SnapBoost.

    Numeric columns are median-imputed, optionally augmented with missingness
    indicators and standardized. Categorical columns are most-frequent-imputed
    and one-hot encoded with unknown-category handling. The returned transformer
    belongs in a normal scikit-learn ``Pipeline`` and does not alter SnapBoost's
    internal training or model format.
    """
    categorical_features = tuple(categorical_features)
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

    encoder_options = {"handle_unknown": "ignore"}
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
