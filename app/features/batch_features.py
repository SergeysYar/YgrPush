from __future__ import annotations

import pandas as pd
from app.data.numeric_parser import parse_numeric


class BatchFeatureBuilder:
    def __init__(self) -> None:
        pass

    def build_full_batch_features(self, batch_df: pd.DataFrame) -> dict[str, float]:
        if batch_df.empty:
            return {
                "num_steps": 0.0,
                "duration_sum": 0.0,
                "average_ph": 0.0,
                "first_ph": 0.0,
                "last_ph": 0.0,
                "min_ph": 0.0,
                "max_ph": 0.0,
                "component_mass_total": 0.0,
                "water_steps": 0.0,
                "salt_steps": 0.0,
                "acid_steps": 0.0,
                "last_completed_step": 0.0,
            }

        duration_sum = 0.0
        if "duration_minutes" in batch_df:
            duration_values = batch_df["duration_minutes"].apply(parse_numeric).dropna()
            duration_sum = float(duration_values.sum()) if not duration_values.empty else 0.0

        ph_values: list[float] = []
        for column in ["startpH", "endpH", "midpH"]:
            if column in batch_df:
                ph_values.extend(
                    value
                    for value in batch_df[column].apply(parse_numeric).tolist()
                    if value is not None
                )

        loading_types = (
            batch_df["loading_step_type"].fillna("").astype(str).str.lower()
            if "loading_step_type" in batch_df
            else pd.Series(dtype=str)
        )
        component_mass_total = 0.0
        for column in [col for col in batch_df.columns if col.startswith("mass_")]:
            component_mass_total += float(
                batch_df[column].apply(parse_numeric).dropna().sum()
            )

        return {
            "num_steps": float(len(batch_df)),
            "duration_sum": duration_sum,
            "average_ph": float(sum(ph_values) / len(ph_values)) if ph_values else 0.0,
            "first_ph": float(ph_values[0]) if ph_values else 0.0,
            "last_ph": float(ph_values[-1]) if ph_values else 0.0,
            "min_ph": float(min(ph_values)) if ph_values else 0.0,
            "max_ph": float(max(ph_values)) if ph_values else 0.0,
            "component_mass_total": component_mass_total,
            "water_steps": float(loading_types.str.contains("water|вода|h2o", regex=True).sum()) if not loading_types.empty else 0.0,
            "salt_steps": float(loading_types.str.contains("salt|соль|nacl|хлорид", regex=True).sum()) if not loading_types.empty else 0.0,
            "acid_steps": float(loading_types.str.contains("acid|кислот|уксус", regex=True).sum()) if not loading_types.empty else 0.0,
            "last_completed_step": float(len(batch_df)),
        }

    def build_checkpoint_features(self, batch_df: pd.DataFrame, checkpoint_order: int | None = None) -> dict[str, float]:
        if checkpoint_order is not None:
            snapshot = batch_df.iloc[:checkpoint_order]
        else:
            snapshot = batch_df
        return self.build_full_batch_features(snapshot)
