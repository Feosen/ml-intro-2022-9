from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore


def create_knn_pipeline(n_neighbors: int) -> Pipeline:
    """Build KNN pipline."""
    pipeline_steps = (
        ("scaler", StandardScaler()),
        (
            "classifier",
            KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1),
        ),
    )
    return Pipeline(steps=pipeline_steps)


def create_rf_pipeline() -> Pipeline:
    """Build Random Forest pipline"""
    pipeline_steps = (
        (
            "classifier",
            RandomForestClassifier(n_jobs=-1),
        ),
    )
    return Pipeline(steps=pipeline_steps)
