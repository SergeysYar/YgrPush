from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Path, Query, Body, Depends

from app.api.dependencies import get_settings
from app.ml.prediction_service import PredictionService
from app.schemas.prediction import (
    BatchPredictionResponse,
    CustomPredictionRequest,
)

router = APIRouter()


@router.post(
    "/predict/batch/{batch_id}",
    summary="Predict quality for batch",
    response_model=BatchPredictionResponse,
)
def predict_batch(
    batch_id: int = Path(..., ge=1),
    up_to_step_order: int | None = Query(default=None, ge=1),
    up_to_measurement_id: int | None = Query(default=None, ge=1),
    settings = Depends(get_settings),
) -> BatchPredictionResponse:
    service = PredictionService(settings.db_path, settings.ml_storage_path)
    try:
        service.train_if_no_model()
        return service.predict_batch(batch_id, up_to_step_order, up_to_measurement_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.post(
    "/predict/custom",
    summary="Predict quality for a custom batch",
    response_model=BatchPredictionResponse,
)
def predict_custom(
    payload: CustomPredictionRequest = Body(...),
    settings = Depends(get_settings),
) -> BatchPredictionResponse:
    service = PredictionService(settings.db_path, settings.ml_storage_path)
    try:
        service.train_if_no_model()
        return service.predict_custom(payload.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
