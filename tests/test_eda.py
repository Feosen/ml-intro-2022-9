from pathlib import Path

from click.testing import CliRunner
import pytest

from ml_intro_2022_9.eda import create_eda


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_success(runner: CliRunner) -> None:
    """Checks successful file creation."""
    with open("tests/train.csv", 'r') as data_file:
        data = data_file.read()

    with runner.isolated_filesystem():
        data_file_name = Path("train.csv")
        with open(data_file_name, 'w') as data_file:
            data_file.write(data)
        eda_file_name = Path("test_eda.html")
        result = runner.invoke(
            create_eda,
            [
                "--dataset-path",
                data_file_name,
                "--save-eda-path",
                eda_file_name,
                "--title",
                "Test title",
            ],
        )
        assert f"EDA is saved to {eda_file_name}." in result.output
        with open(eda_file_name, "r") as file:
            assert "<title>Test title</title>" in file.read()
