from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(n_neighbors: bool) -> Pipeline:
    pipeline_steps = (
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors,
                                            n_jobs=-1)),
    )
    return Pipeline(steps=pipeline_steps)
