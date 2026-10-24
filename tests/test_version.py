import subprocess
import sys

from mercari_shopper import __version__
from mercari_shopper.cli import main


def test_version_string() -> None:
    assert __version__ == "0.1.0"


def test_cli_prints_version() -> None:
    import io

    stdout = io.StringIO()
    assert main(["--version"], stdout=stdout) == 0
    assert stdout.getvalue() == "mercari-ai-shopper 0.1.0\n"


def test_module_entry_point() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mercari_shopper", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "mercari-ai-shopper 0.1.0" in result.stdout
