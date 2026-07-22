from __future__ import annotations

from fastapi import FastAPI

from .routes.health import router as health_router
from .routes.batches import router as batches_router
from .routes.data import router as data_router
from .routes.models import router as models_router
from .routes.predict import router as predict_router
from .routes.predictions import router as predictions_router
from .routes.reports import router as reports_router
from .routes.train import router as train_router

app = FastAPI(title="Shampoo Quality Forecast API", version="0.1.0")

app.include_router(health_router, prefix="/health")
app.include_router(data_router, prefix="/api/v1/data")
app.include_router(batches_router, prefix="/api/v1/batches")
app.include_router(models_router, prefix="/api/v1/models")
app.include_router(train_router, prefix="/api/v1")
app.include_router(predict_router, prefix="/api/v1")
app.include_router(predictions_router, prefix="/api/v1/predictions")
app.include_router(reports_router, prefix="/api/v1/reports")
