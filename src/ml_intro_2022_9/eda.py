from pathlib import Path
import pandas as pd  # type: ignore
from pandas_profiling import ProfileReport  # type: ignore

import click


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
    "--save-eda-path",
    default="data/eda.html",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--title",
    default="ML intro 2022 task 9 Profiling Report",
    type=str,
    show_default=True,
)
@click.option(
    "--explorative",
    default=True,
    type=bool,
    show_default=True,
)
def create_eda(
    dataset_path: Path,
    save_eda_path: Path,
    title: str,
    explorative: bool,
) -> None:
    data = pd.read_csv(dataset_path)
    profile = ProfileReport(data, title=title, explorative=explorative)
    profile.to_file(save_eda_path)
    click.echo(f"EDA saved to {save_eda_path}.")
