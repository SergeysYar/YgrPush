from __future__ import annotations

import pandas as pd
from app.data.numeric_parser import parse_numeric


class BatchFeatureBuilder:
    def __init__(self) -> None:
        pass

    def _numeric_series(self, batch_df: pd.DataFrame, column: str) -> pd.Series:
        if column not in batch_df:
            return pd.Series(dtype=float)
        values = batch_df[column].apply(parse_numeric).dropna()
        return values.astype(float) if not values.empty else pd.Series(dtype=float)

    def _sum_mass_columns(self, batch_df: pd.DataFrame) -> pd.Series:
        total = pd.Series(0.0, index=batch_df.index, dtype=float)
        mass_columns = [col for col in batch_df.columns if col.startswith("mass_")]
        for column in mass_columns:
            total = total.add(
                batch_df[column].apply(lambda value: parse_numeric(value) or 0.0),
                fill_value=0.0,
            )
        return total

    def _count_non_null_components(self, batch_df: pd.DataFrame) -> int:
        component_values: set[int | str] = set()
        for column in [col for col in batch_df.columns if col.startswith("component_")]:
            for value in batch_df[column].dropna().tolist():
                if value != "":
                    component_values.add(value)
        return len(component_values)

    def build_full_batch_features(self, batch_df: pd.DataFrame) -> dict[str, float]:
        if batch_df.empty:
            return {
                "num_steps": 0.0,
                "duration_sum": 0.0,
                "avg_step_duration": 0.0,
                "max_step_duration": 0.0,
                "invalid_durations_count": 0.0,
                "average_ph": 0.0,
                "first_ph": 0.0,
                "last_ph": 0.0,
                "min_ph": 0.0,
                "max_ph": 0.0,
                "avg_temp": 0.0,
                "min_temp": 0.0,
                "max_temp": 0.0,
                "first_temp": 0.0,
                "last_temp": 0.0,
                "avg_freq": 0.0,
                "max_freq": 0.0,
                "avg_pe": 0.0,
                "max_pe": 0.0,
                "component_mass_total": 0.0,
                "unique_components_count": 0.0,
                "water_mass_total": 0.0,
                "salt_mass_total": 0.0,
                "acid_mass_total": 0.0,
                "water_steps": 0.0,
                "salt_steps": 0.0,
                "acid_steps": 0.0,
                "measurement_steps": 0.0,
                "correction_steps": 0.0,
                "missing_sensor_readings": 0.0,
                "suspicious_values_count": 0.0,
                "last_completed_step": 0.0,
            }

        duration_values = self._numeric_series(batch_df, "duration_minutes")
        duration_sum = float(duration_values.sum()) if not duration_values.empty else 0.0
        avg_step_duration = float(duration_values.mean()) if not duration_values.empty else 0.0
        max_step_duration = float(duration_values.max()) if not duration_values.empty else 0.0
        invalid_durations_count = float((duration_values < 0).sum()) if not duration_values.empty else 0.0

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
        row_mass_total = self._sum_mass_columns(batch_df)
        component_mass_total = float(row_mass_total.sum())

        temp_values = pd.concat(
            [
                self._numeric_series(batch_df, "startTemp"),
                self._numeric_series(batch_df, "endTemp"),
                self._numeric_series(batch_df, "midTemp"),
            ],
            ignore_index=True,
        )
        freq_values = pd.concat(
            [
                self._numeric_series(batch_df, "startfreq"),
                self._numeric_series(batch_df, "endfreq"),
                self._numeric_series(batch_df, "midfreq"),
            ],
            ignore_index=True,
        )
        pe_values = pd.concat(
            [
                self._numeric_series(batch_df, "startPE"),
                self._numeric_series(batch_df, "endPE"),
                self._numeric_series(batch_df, "midPE"),
            ],
            ignore_index=True,
        )

        water_mask = loading_types.str.contains("water|вода|h2o", regex=True) if not loading_types.empty else pd.Series(False, index=batch_df.index)
        salt_mask = loading_types.str.contains("salt|соль|nacl|хлорид", regex=True) if not loading_types.empty else pd.Series(False, index=batch_df.index)
        acid_mask = loading_types.str.contains("acid|кислот|уксус", regex=True) if not loading_types.empty else pd.Series(False, index=batch_df.index)
        measurement_mask = loading_types.str.contains("measure|измер", regex=True) if not loading_types.empty else pd.Series(False, index=batch_df.index)
        correction_mask = loading_types.str.contains("correct|коррект", regex=True) if not loading_types.empty else pd.Series(False, index=batch_df.index)

        sensor_columns = [
            column
            for column in ["startTemp", "endTemp", "startpH", "endpH", "startPE", "endPE", "startfreq", "endfreq"]
            if column in batch_df
        ]
        missing_sensor_readings = float(batch_df[sensor_columns].isna().sum().sum()) if sensor_columns else 0.0
        suspicious_values_count = float(sum(1 for value in ph_values if value < 0 or value > 14))

        return {
            "num_steps": float(len(batch_df)),
            "duration_sum": duration_sum,
            "avg_step_duration": avg_step_duration,
            "max_step_duration": max_step_duration,
            "invalid_durations_count": invalid_durations_count,
            "average_ph": float(sum(ph_values) / len(ph_values)) if ph_values else 0.0,
            "first_ph": float(ph_values[0]) if ph_values else 0.0,
            "last_ph": float(ph_values[-1]) if ph_values else 0.0,
            "min_ph": float(min(ph_values)) if ph_values else 0.0,
            "max_ph": float(max(ph_values)) if ph_values else 0.0,
            "avg_temp": float(temp_values.mean()) if not temp_values.empty else 0.0,
            "min_temp": float(temp_values.min()) if not temp_values.empty else 0.0,
            "max_temp": float(temp_values.max()) if not temp_values.empty else 0.0,
            "first_temp": float(temp_values.iloc[0]) if not temp_values.empty else 0.0,
            "last_temp": float(temp_values.iloc[-1]) if not temp_values.empty else 0.0,
            "avg_freq": float(freq_values.mean()) if not freq_values.empty else 0.0,
            "max_freq": float(freq_values.max()) if not freq_values.empty else 0.0,
            "avg_pe": float(pe_values.mean()) if not pe_values.empty else 0.0,
            "max_pe": float(pe_values.max()) if not pe_values.empty else 0.0,
            "component_mass_total": component_mass_total,
            "unique_components_count": float(self._count_non_null_components(batch_df)),
            "water_mass_total": float(row_mass_total[water_mask].sum()) if len(row_mass_total) else 0.0,
            "salt_mass_total": float(row_mass_total[salt_mask].sum()) if len(row_mass_total) else 0.0,
            "acid_mass_total": float(row_mass_total[acid_mask].sum()) if len(row_mass_total) else 0.0,
            "water_steps": float(water_mask.sum()) if not loading_types.empty else 0.0,
            "salt_steps": float(salt_mask.sum()) if not loading_types.empty else 0.0,
            "acid_steps": float(acid_mask.sum()) if not loading_types.empty else 0.0,
            "measurement_steps": float(measurement_mask.sum()) if not loading_types.empty else 0.0,
            "correction_steps": float(correction_mask.sum()) if not loading_types.empty else 0.0,
            "missing_sensor_readings": missing_sensor_readings,
            "suspicious_values_count": suspicious_values_count,
            "last_completed_step": float(len(batch_df)),
        }

    def build_checkpoint_features(self, batch_df: pd.DataFrame, checkpoint_order: int | None = None) -> dict[str, float]:
        if checkpoint_order is not None:
            snapshot = batch_df.iloc[:checkpoint_order]
        else:
            snapshot = batch_df
        return self.build_full_batch_features(snapshot)
