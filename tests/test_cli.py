import subprocess
import sys


def test_cli_inspect_db_help():
    result = subprocess.run([sys.executable, "-m", "app.cli", "inspect-db"], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout is not None
