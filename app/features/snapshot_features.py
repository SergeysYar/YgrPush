from __future__ import annotations

import pandas as pd
from typing import Any

from app.features.batch_features import BatchFeatureBuilder


class SnapshotBuilder:
    """Create snapshots of batch state at each step for intermediate predictions."""

    def build_batch_snapshots(self, batch_df: pd.DataFrame, batch_id: int, target_values: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Build snapshots for each step in a batch.
        
        Returns list of dicts with:
        - batch_id
        - checkpoint_order
        - completed_steps
        - is_final_snapshot
        - features (dict from BatchFeatureBuilder)
        - targets (same for all snapshots of one batch)
        """
        snapshots = []
        feature_builder = BatchFeatureBuilder()

        for idx in range(len(batch_df)):
            completed_steps = idx + 1
            features = feature_builder.build_checkpoint_features(
                batch_df, checkpoint_order=completed_steps
            )
            snapshot = {
                "batch_id": batch_id,
                "checkpoint_order": completed_steps,
                "completed_steps": completed_steps,
                "is_final_snapshot": idx == len(batch_df) - 1,
                "features": features,
                "targets": target_values or {}
            }
            snapshots.append(snapshot)

        return snapshots
