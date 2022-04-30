from pathlib import Path
from typing import Tuple, Any

import click
import mlflow  # type: ignore
import numpy as np
from joblib import dump  # type: ignore
from sklearn.metrics import f1_score  # type: ignore
from sklearn.model_selection import (  # type: ignore
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)


from .data import get_dataset
from .git import get_git_revision_hash
from .pipeline import create_tree_pipeline, create_knn_pipeline


def _process_linspace_args(x: Tuple[Any, Any, Any]) -> Tuple[Any, Any, int]:
    return x[0], x[1], int(x[2])


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
    "--knn-n-estimators",
    default=2,
    type=click.IntRange(0, min_open=True),
    show_default=True,
)
@click.option(
    "--knn-n-neighbors",
    nargs=3,
    type=click.IntRange(0, min_open=True),
)
@click.option(
    "--knn-max-features",
    nargs=3,
    type=click.FloatRange(0, min_open=True),
)
@click.option(
    "--tree-n-estimators",
    default=100,
    type=click.IntRange(0, min_open=True),
    show_default=True,
)
@click.option(
    "--tree-max-depth",
    nargs=3,
    type=click.IntRange(0, min_open=True),
    show_default=True,
)
@click.option(
    "--tree-max-depth-none",
    type=bool,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    data_size: float,
    random_state: int,
    cv: int,
    knn_n_estimators: int,
    knn_n_neighbors: Tuple[int, int, int],
    knn_max_features: Tuple[float, float, float],
    tree_n_estimators: int,
    tree_max_depth: Tuple[int, int, int],
    tree_max_depth_none: bool,
) -> None:
    with mlflow.start_run():
        mlflow.log_param("git_rev_hash", get_git_revision_hash())
        mlflow.log_param("data_size", data_size)
        mlflow.log_param("random_state", random_state)

        mlflow.log_param("knn_n_estimators", knn_n_estimators)
        knn_n_neighbors_space = np.linspace(
            *_process_linspace_args(knn_n_neighbors), dtype=np.uint
        )
        mlflow.log_param("knn_n_neighbors_space", knn_n_neighbors_space)
        knn_max_features_space = np.linspace(
            *_process_linspace_args(knn_max_features)
        )
        mlflow.log_param("knn_max_features_space", knn_max_features_space)

        pgrid_knn = [
            {
                "clf__base_estimator__n_neighbors": knn_n_neighbors_space,
                "clf__max_features": knn_max_features_space,
            }
        ]

        mlflow.log_param("tree_n_estimators", tree_n_estimators)
        tree_max_depth_space = np.linspace(
            *_process_linspace_args(tree_max_depth), dtype=np.uint
        ).tolist()
        if tree_max_depth_none:
            tree_max_depth_space += [None]
        mlflow.log_param("tree_max_depth_space", tree_max_depth_space)
        tree_criterion_space = ["gini", "entropy"]
        mlflow.log_param("tree_criterion_space", tree_criterion_space)

        pgrid_tree = [
            {
                "clf__max_depth": tree_max_depth_space,
                "clf__criterion": tree_criterion_space,
            }
        ]

        knn_pipeline = create_knn_pipeline(
            n_estimators=knn_n_estimators,
            n_neighbors=5,
            max_features=1.0,
            random_state=random_state,
        )
        tree_pipeline = create_tree_pipeline(
            n_estimators=tree_n_estimators,
            max_depth=None,
            criterion="gini",
            random_state=random_state,
        )

        gridcvs = {}
        inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
        for name, pgrid, est in (
            ("knn", pgrid_knn, knn_pipeline),
            ("tree", pgrid_tree, tree_pipeline),
        ):
            gcv = GridSearchCV(
                estimator=est,
                param_grid=pgrid,
                scoring="f1_macro",
                n_jobs=-1,
                cv=inner_cv,
                verbose=0,
                refit=True,
            )
            gridcvs[name] = gcv

        features, target = get_dataset(dataset_path, random_state, data_size)
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, train_size=0.8, random_state=1, stratify=target
        )

        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        nested_scores = []
        for name, gs_est in sorted(gridcvs.items()):
            nested_score = cross_val_score(
                gs_est, X=X_train, y=y_train, cv=outer_cv, n_jobs=1
            )
            nested_scores.append((nested_score.mean(), name))
            click.echo(
                f"{name} | score {nested_score.mean() * 100:.2f}"
                + f" +/- {nested_score.std() * 100:.2f}"
            )
        nested_scores.sort()

        best_pipeline_name = nested_scores[-1][1]
        best_cvs = gridcvs[best_pipeline_name]
        best_cvs.fit(X_train, y_train)
        best_pipeline = best_cvs.best_estimator_
        best_params = best_cvs.best_params_

        if best_pipeline_name == "tree":
            mlflow.log_param(
                "best_tree_max_depth", best_params["clf__max_depth"]
            )
            mlflow.log_param(
                "best_tree_criterion", best_params["clf__criterion"]
            )
        elif best_pipeline_name == "knn":
            mlflow.log_param(
                "best_knn_n_neighbors",
                best_params["clf__base_estimator__n_neighbors"],
            )
            mlflow.log_param(
                "best_knn_max_features", best_params["clf__max_features"]
            )
        else:
            raise ValueError(best_pipeline_name)

        scoring = ["weighted", "micro", "macro"]
        for score_name in scoring:
            train_score = f1_score(
                y_true=y_train,
                y_pred=best_pipeline.predict(X_train),
                average=score_name,
            )
            test_score = f1_score(
                y_true=y_test,
                y_pred=best_pipeline.predict(X_test),
                average=score_name,
            )
            click.echo(f"train f1_{score_name}: {train_score:.4f}")
            click.echo(f"test f1_{score_name}: {test_score:.4f}")
            mlflow.log_metric(f"train f1_{score_name}", train_score)
            mlflow.log_metric(f"test f1_{score_name}", test_score)

        mlflow.sklearn.log_model(best_pipeline, "model")
        best_pipeline.fit(features, target)
        dump(best_pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
