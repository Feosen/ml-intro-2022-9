import pprint
from pathlib import Path

import click
from joblib import load  # type: ignore


@click.command()
@click.option(
    "-m",
    "--model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
def inspect(
    model_path: Path,
) -> None:
    pipeline = load(model_path)
    click.echo(pprint.pformat(pipeline.get_params(deep=True)))
