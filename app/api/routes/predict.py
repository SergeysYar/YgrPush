from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query

router = APIRouter()


@router.post("/predict/batch/{batch_id}", summary="Predict quality for batch")
def predict_batch(
    batch_id: int = Path(..., ge=1),
    up_to_step_order: int | None = Query(default=None, ge=1),
    up_to_measurement_id: int | None = Query(default=None, ge=1),
) -> dict[str, object]:
    return {
        "batch_id": batch_id,
        "checkpoint": {
            "completed_steps": up_to_step_order or 0,
            "last_measurement_id": up_to_measurement_id or 0,
        },
        "data_quality": {
            "missing_count": 0,
            "invalid_count": 0,
            "is_anomalous": False,
            "warnings": [],
        },
        "predictions": {
            "ph": {},
            "viscosity": {},
            "chlorides": {},
        },
        "similar_batches": [],
    }


@router.post("/predict/custom", summary="Predict quality for a custom batch")
def predict_custom(payload: dict[str, object]) -> dict[str, object]:
    return {"status": "not_implemented", "payload": payload}
