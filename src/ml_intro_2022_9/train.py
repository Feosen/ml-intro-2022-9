from enum import Enum
from pathlib import Path
from typing import List

import click
import mlflow  # type: ignore
import numpy as np  # type: ignore
from joblib import dump  # type: ignore
from sklearn.model_selection import cross_validate  # type: ignore

from .data import get_dataset
from .git import get_git_revision_hash
from .pipeline import create_rf_pipeline, create_knn_pipeline


class ModelName(Enum):
    knn = "knn"
    random_forest = "random-forest"

    @classmethod
    def values(cls) -> List[str]:
        return [v.value for v in cls]


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "-m",
    "--model-name",
    default="knn",
    type=click.Choice(ModelName.values()),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--cv",
    default=5,
    type=click.IntRange(0, min_open=True),
    show_default=True,
)
@click.option(
    "--n-neighbors",
    default=5,
    type=click.IntRange(0, min_open=True),
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    model_name: str,
    random_state: int,
    cv: int,
    n_neighbors: int,
) -> None:
    features, target = get_dataset(dataset_path, random_state)
    with mlflow.start_run():
        mlflow.log_param("git_rev_hash", get_git_revision_hash())
        mlflow.log_param("model_name", model_name)
        if model_name == ModelName.knn.value:
            pipeline = create_knn_pipeline(n_neighbors=n_neighbors)
            mlflow.log_param("n_neighbors", n_neighbors)
        elif model_name == ModelName.random_forest.value:
            pipeline = create_rf_pipeline()

        scoring = ["accuracy", "f1_micro", "f1_macro"]
        results = cross_validate(
            pipeline,
            features,
            target,
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
            scoring=scoring,
        )
        for score_name in scoring:
            for data_type in ("train", "test"):
                score_full_name = f"{data_type}_{score_name}"
                score_value = float(np.mean(results[score_full_name]))
                click.echo(f"{score_full_name}: {score_value:.4f}")
                mlflow.log_metric(score_full_name, score_value)

        mlflow.sklearn.log_model(pipeline, "model")
        pipeline.fit(features, target)
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
