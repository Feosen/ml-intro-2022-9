from pathlib import Path
from unittest import mock

from click.testing import CliRunner
import pytest
from joblib import load

from ml_intro_2022_9.train import train
from sklearn.pipeline import Pipeline


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_always_error(runner: CliRunner) -> None:
    """It fails when random state has wrong data type."""
    result = runner.invoke(
        train,
        [
            "--random-state",
            "trash",
        ],
    )
    assert result.exit_code == 2
    assert (
        "Invalid value for '--random-state': 'trash' is not a valid integer."
        in result.output
    )


def test_tree_success(runner: CliRunner) -> None:
    """Checks successful file creation."""
    with open("tests/train.csv", "r") as data_file:
        data = data_file.read()

    with runner.isolated_filesystem():
        data_file_name = Path("train.csv")
        with open(data_file_name, "w") as data_file:
            data_file.write(data)
        model_file_name = Path("model.joblib")

        with mock.patch(
            "ml_intro_2022_9.train.get_git_revision_hash",
            lambda: "ed664c85a14907406257a3218c985d2d48b70de0",
        ):
            result = runner.invoke(
                train,
                [
                    "--dataset-path",
                    data_file_name,
                    "--save-model-path",
                    model_file_name,
                    "--model-name",
                    "tree",
                    "--cv",
                    2,
                ],
            )
        assert f"Model is saved to {model_file_name}." in result.output
        model = load(model_file_name)
        assert isinstance(model, Pipeline)


def test_knn_success(runner: CliRunner) -> None:
    """Checks successful file creation."""
    with open("tests/train.csv", "r") as data_file:
        data = data_file.read()

    with runner.isolated_filesystem():
        data_file_name = Path("train.csv")
        with open(data_file_name, "w") as data_file:
            data_file.write(data)
        model_file_name = Path("model.joblib")

        with mock.patch(
            "ml_intro_2022_9.train.get_git_revision_hash",
            lambda: "ed664c85a14907406257a3218c985d2d48b70de0",
        ):
            result = runner.invoke(
                train,
                [
                    "--dataset-path",
                    data_file_name,
                    "--save-model-path",
                    model_file_name,
                    "--model-name",
                    "knn",
                    "--knn-n-neighbors",
                    2,
                    "--cv",
                    2,
                ],
            )
        assert f"Model is saved to {model_file_name}." in result.output
        model = load(model_file_name)
        assert isinstance(model, Pipeline)
