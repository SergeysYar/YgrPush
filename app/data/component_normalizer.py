from __future__ import annotations

from typing import Any
import pandas as pd

# Типы этапов варки
STEP_TYPE_MAPPING = {
    "water": ["water", "вода", "h2o"],
    "salt": ["salt", "соль", "nacl", "хлорид"],
    "acid": ["acid", "кислота", "уксусная", "acetic"],
    "surfactant": ["surf", "пав", "surfactant", "detergent"],
    "thickener": ["thick", "загуститель", "xanthan"],
    "preservative": ["pres", "консервант", "phenoxyethanol"],
    "fragrance": ["frag", "отдушка", "aroma"],
    "colorant": ["color", "краситель", "dye"],
    "other": []
}

def normalize_step_type(step_type: str | None) -> str:
    """Normalize loading step type to category."""
    if step_type is None:
        return "other"
    normalized = step_type.lower().strip()
    for category, keywords in STEP_TYPE_MAPPING.items():
        if category != "other" and any(kw in normalized for kw in keywords):
            return category
    return "other"


class ComponentNormalizer:
    """Convert wide component format (component_1, mass_1, ...) to long format."""

    def __init__(self, max_components: int = 7) -> None:
        self.max_components = max_components
        self.component_cols = [f"component_{i}" for i in range(1, max_components + 1)]
        self.mass_cols = [f"mass_{i}" for i in range(1, max_components + 1)]

    def normalize_batch_components(self, batch_df: pd.DataFrame) -> pd.DataFrame:
        """Convert component data from wide to long format for a batch.
        
        Returns DataFrame with columns:
        - measurement_id
        - step_order
        - component_position
        - component_id
        - component_mass
        """
        rows = []
        for idx, row in batch_df.iterrows():
            for pos in range(1, self.max_components + 1):
                comp_col = f"component_{pos}"
                mass_col = f"mass_{pos}"
                if comp_col in row and mass_col in row:
                    comp_id = row[comp_col]
                    mass = row[mass_col]
                    if comp_id is not None and comp_id != "" and mass is not None:
                        rows.append({
                            "measurement_id": row.get("id"),
                            "step_order": idx,
                            "component_position": pos,
                            "component_id": int(comp_id) if isinstance(comp_id, (int, float)) else None,
                            "component_mass": float(mass) if isinstance(mass, (int, float)) else None,
                        })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def aggregate_components_by_batch(self, components_df: pd.DataFrame) -> dict[str, Any]:
        """Aggregate component data across a full batch.
        
        Returns dict with:
        - total_mass_by_component
        - count_appearances
        - first_step_by_component
        - last_step_by_component
        - mass_by_category
        """
        if components_df.empty:
            return {}

        agg = {
            "total_mass_by_component": {},
            "count_appearances": {},
            "first_step": {},
            "last_step": {},
            "mass_by_category": {}
        }

        for comp_id in components_df["component_id"].unique():
            comp_data = components_df[components_df["component_id"] == comp_id]
            agg["total_mass_by_component"][int(comp_id)] = float(comp_data["component_mass"].sum())
            agg["count_appearances"][int(comp_id)] = len(comp_data)
            agg["first_step"][int(comp_id)] = int(comp_data["step_order"].min())
            agg["last_step"][int(comp_id)] = int(comp_data["step_order"].max())

        return agg
