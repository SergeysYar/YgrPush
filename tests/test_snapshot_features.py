import pandas as pd

from app.features.snapshot_features import SnapshotBuilder


def test_snapshot_builder_uses_completed_steps():
    batch_df = pd.DataFrame(
        [
            {"id": 1, "duration_minutes": 10.0, "startpH": 5.5, "endpH": 5.7},
            {"id": 2, "duration_minutes": 20.0, "startpH": 5.7, "endpH": 5.9},
        ]
    )

    snapshots = SnapshotBuilder().build_batch_snapshots(batch_df, batch_id=101)

    assert len(snapshots) == 2
    assert snapshots[0]["completed_steps"] == 1
    assert snapshots[0]["features"]["num_steps"] == 1.0
    assert snapshots[0]["features"]["duration_sum"] == 10.0
    assert snapshots[1]["features"]["num_steps"] == 2.0
    assert snapshots[1]["is_final_snapshot"] is True
