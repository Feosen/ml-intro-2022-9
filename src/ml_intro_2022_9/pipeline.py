from typing import Union, Any

import pandas as pd  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.ensemble import (  # type: ignore
    RandomForestClassifier,
    BaggingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore


def select_useless(x: pd.DataFrame) -> Any:
    return x.columns[x.nunique(axis=0) == 1].to_list()


def select_non_bin(x: pd.DataFrame) -> Any:
    return x.columns[x.nunique(axis=0) > 2].to_list()


def _create_data_transformer() -> BaseEstimator:
    return ColumnTransformer(
        [
            ("drop_useless", "drop", select_useless),
            ("scale_non_bin", StandardScaler(), select_non_bin),
        ],
        remainder="passthrough",
        n_jobs=-1,
    )


def create_knn_pipeline(
    n_neighbors: int,
    n_estimators: int,
    max_samples: Union[int, float],
    max_features: Union[int, float],
) -> Pipeline:
    """Build KNN pipline."""

    transformer = _create_data_transformer()

    clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)

    meta_clf = BaggingClassifier(
        clf,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        n_jobs=-1,
    )

    pipeline_steps = (
        ("transformer", transformer),
        ("classifier", meta_clf),
    )
    return Pipeline(steps=pipeline_steps)


def create_rf_pipeline(
    n_estimators: int,
    max_depth: Union[None, int],
    min_samples_split: Union[int, float],
    min_samples_leaf: Union[int, float],
) -> Pipeline:
    """Build Random Forest pipline"""

    transformer = _create_data_transformer()

    meta_clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
    )

    pipeline_steps = (
        ("transformer", transformer),
        ("classifier", meta_clf),
    )
    return Pipeline(steps=pipeline_steps)
