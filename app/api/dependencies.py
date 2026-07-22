from __future__ import annotations

from fastapi import Depends

from app.config import settings


def get_settings() -> type(settings):
    return settings
