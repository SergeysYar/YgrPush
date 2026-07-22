from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    db_path: Path = Field(default=Path("./production.db"), validation_alias="DB_PATH")
    ml_storage_path: Path = Field(default=Path("./ml_storage.db"), validation_alias="ML_STORAGE_PATH")
    target_protocol_policy: Literal["latest", "first", "all"] = Field(
        default="latest",
        validation_alias="TARGET_PROTOCOL_POLICY",
    )
    catboost_min_labeled_batches: int = Field(
        default=40,
        validation_alias="CATBOOST_MIN_LABELED_BATCHES",
    )
    anomaly_min_batches: int = Field(default=40, validation_alias="ANOMALY_MIN_BATCHES")
    confidence_high_min_batches: int = Field(
        default=20,
        validation_alias="CONFIDENCE_HIGH_MIN_BATCHES",
    )
    max_other_components: int = Field(default=5, validation_alias="MAX_OTHER_COMPONENTS")
    step_frequency_threshold: float = Field(
        default=20.0,
        validation_alias="STEP_FREQUENCY_THRESHOLD",
    )
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
