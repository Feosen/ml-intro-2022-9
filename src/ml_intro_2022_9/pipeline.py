from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore


def create_pipeline(n_neighbors: int) -> Pipeline:
    pipeline_steps = (
        ("scaler", StandardScaler()),
        (
            "classifier",
            KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1),
        ),
    )
    return Pipeline(steps=pipeline_steps)
