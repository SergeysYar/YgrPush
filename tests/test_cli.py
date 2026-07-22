import subprocess
import sys
import types

from app import cli


def test_cli_inspect_db_help():
    result = subprocess.run([sys.executable, "-m", "app.cli", "inspect-db"], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout is not None


def test_cli_train_forwards_protocol_policy_and_model_types(monkeypatch):
    captured: dict[str, object] = {}

    class DummyPipeline:
        def __init__(self, db_path, ml_storage_path):
            captured["db_path"] = db_path
            captured["ml_storage_path"] = ml_storage_path

        def train_all(self, snapshot_mode=False, protocol_policy=None, model_types=None):
            captured["snapshot_mode"] = snapshot_mode
            captured["protocol_policy"] = protocol_policy
            captured["model_types"] = model_types
            return {}

        def save_cv_report(self, results):
            captured["saved"] = results

    dummy_module = types.ModuleType("app.ml.service")
    dummy_module.TrainingPipeline = DummyPipeline
    monkeypatch.setitem(sys.modules, "app.ml.service", dummy_module)

    exit_code = cli.main(
        [
            "train",
            "--model-types",
            "baseline",
            "ridge",
            "--protocol-policy",
            "first",
            "--batch-mode",
        ]
    )

    assert exit_code == 0
    assert captured["snapshot_mode"] is False
    assert captured["protocol_policy"] == "first"
    assert captured["model_types"] == ["baseline", "ridge"]
    assert captured["saved"] == {}
