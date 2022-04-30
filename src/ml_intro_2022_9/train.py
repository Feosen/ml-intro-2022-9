from enum import Enum
from pathlib import Path
from typing import List, Union

import click
import mlflow  # type: ignore
import numpy as np
from joblib import dump  # type: ignore
from sklearn.model_selection import cross_validate  # type: ignore

from .data import get_dataset
from .git import get_git_revision_hash
from .pipeline import create_rf_pipeline, create_knn_pipeline


class ModelName(Enum):
    knn = "knn"
    tree = "tree"

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
    "--data-size",
    default=1,
    type=click.FloatRange(0, 1, min_open=True),
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
    "--n-estimators",
    default=100,
    type=click.IntRange(0, min_open=True),
    show_default=True,
)
@click.option(
    "--knn-n-neighbors",
    default=5,
    type=click.IntRange(0, min_open=True),
    show_default=True,
)
@click.option(
    "--knn-max-samples",
    default=1,
    type=click.FloatRange(0, 1, min_open=True),
    show_default=True,
)
@click.option(
    "--knn-max-features",
    default=1,
    type=click.FloatRange(0, 1, min_open=True),
    show_default=True,
)
@click.option(
    "--tree-max-depth",
    default=None,
    type=int,
    show_default=True,
)
@click.option(
    "--tree-min-samples-split",
    default=2,
    type=click.IntRange(0, min_open=True),
    show_default=True,
)
@click.option(
    "--tree-min-samples-leaf",
    default=1,
    type=click.IntRange(0, min_open=True),
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    data_size: float,
    model_name: str,
    random_state: int,
    cv: int,
    n_estimators: int,
    knn_n_neighbors: int,
    knn_max_samples: float,
    knn_max_features: float,
    tree_max_depth: Union[None, int],
    tree_min_samples_split: Union[int, float],
    tree_min_samples_leaf: Union[int, float],
) -> None:
    with mlflow.start_run():
        mlflow.log_param("git_rev_hash", get_git_revision_hash())
        mlflow.log_param("data_size", data_size)
        features, target = get_dataset(dataset_path, random_state, data_size)
        mlflow.log_param("model_name", model_name)
        if model_name == ModelName.knn.value:
            mlflow.log_param("knn_n_neighbors", knn_n_neighbors)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("knn_max_samples", knn_max_samples)
            mlflow.log_param("knn_max_features", knn_max_features)
            pipeline = create_knn_pipeline(
                n_neighbors=knn_n_neighbors,
                n_estimators=n_estimators,
                max_samples=knn_max_samples,
                max_features=knn_max_features,
            )
        elif model_name == ModelName.tree.value:
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("tree_max_depth", tree_max_depth)
            mlflow.log_param("tree_min_samples_split", tree_min_samples_split)
            mlflow.log_param("tree_min_samples_leaf", tree_min_samples_leaf)
            pipeline = create_rf_pipeline(
                n_estimators=n_estimators,
                max_depth=tree_max_depth,
                min_samples_split=tree_min_samples_split,
                min_samples_leaf=tree_min_samples_leaf,
            )

        scoring = ["f1_weighted", "f1_micro", "f1_macro"]
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
