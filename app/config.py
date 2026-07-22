from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    db_path: Path = Field(default=Path("./production.db"), env="DB_PATH")
    ml_storage_path: Path = Field(default=Path("./ml_storage.db"), env="ML_STORAGE_PATH")
    target_protocol_policy: Literal["latest", "first", "all"] = Field(default="latest", env="TARGET_PROTOCOL_POLICY")
    catboost_min_labeled_batches: int = Field(default=40, env="CATBOOST_MIN_LABELED_BATCHES")
    anomaly_min_batches: int = Field(default=40, env="ANOMALY_MIN_BATCHES")
    confidence_high_min_batches: int = Field(default=20, env="CONFIDENCE_HIGH_MIN_BATCHES")
    max_other_components: int = Field(default=5, env="MAX_OTHER_COMPONENTS")
    step_frequency_threshold: float = Field(default=20.0, env="STEP_FREQUENCY_THRESHOLD")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

settings = Settings()
