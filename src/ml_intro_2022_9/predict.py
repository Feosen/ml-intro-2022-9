from pathlib import Path

import click
import pandas as pd  # type: ignore
from joblib import load  # type: ignore

from .data import get_test_dataset


@click.command()
@click.option(
    "-d",
    "--test-path",
    default="data/test.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-m",
    "--model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--predict-path",
    default="data/predict.csv",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
def predict(
    test_path: Path,
    model_path: Path,
    predict_path: Path,
) -> None:
    pipeline = load(model_path)
    test_dataset = get_test_dataset(test_path)
    y = pipeline.predict(test_dataset)
    result = pd.DataFrame(
        index=test_dataset.index, data=y, columns=["Cover_Type"]
    )
    result.to_csv(predict_path)
    click.echo(f"Predict is saved to {predict_path}.")
