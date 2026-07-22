from __future__ import annotations

import pandas as pd


class BatchFeatureBuilder:
    def __init__(self) -> None:
        pass

    def build_full_batch_features(self, batch_df: pd.DataFrame) -> dict[str, float]:
        return {
            "num_steps": float(len(batch_df)),
            "duration_sum": float(batch_df["duration_minutes"].dropna().sum()) if "duration_minutes" in batch_df else 0.0,
            "average_ph": float(batch_df[["startpH", "endpH"]].stack().dropna().astype(float).mean()) if any(col in batch_df for col in ["startpH", "endpH"]) else 0.0,
        }

    def build_checkpoint_features(self, batch_df: pd.DataFrame, checkpoint_order: int | None = None) -> dict[str, float]:
        if checkpoint_order is not None:
            snapshot = batch_df.iloc[:checkpoint_order]
        else:
            snapshot = batch_df
        return self.build_full_batch_features(snapshot)
