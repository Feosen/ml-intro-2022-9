from pathlib import Path
from typing import Tuple

import click
import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore


def get_dataset(
    csv_path: Path, random_state: int, data_size: float
) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    # I need to reduce amount of data because of hardware limitation.
    # It's learning task, not commercial.
    if data_size < 1:
        features, _, target, _ = train_test_split(
            features, target, test_size=data_size, random_state=random_state
        )
    return features, target
