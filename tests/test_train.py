from click.testing import CliRunner
import pytest

from ml_intro_2022_9.train import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_always_error(runner: CliRunner) -> None:
    """It fails when test split ratio is greater than 1."""
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
