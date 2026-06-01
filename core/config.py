#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
from pydantic import BaseModel

dotenv_path = ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)


def _env(key: str, default=None, *, required=True):
    """读取环境变量， 且未设置时直接报错，避免 str(None) 问题"""
    val = os.getenv(key, default)
    if required and val is None:
        raise ValueError(f"环境变量 {key} 未设置")
    return val


class AppConfig(BaseModel):
    api_key: str
    base_url: str
    chat_model: str
    summary_model: str
    db_type: str
    llm_temperature: float
    llm_timeout: int
    firecrawl_api_key: str
    mongodb_session_url: str | None = None

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            api_key=_env("API_KEY"),
            base_url=_env("BASE_URL"),
            chat_model=_env("CHAT_MODEL"),
            summary_model=_env("SUMMARY_MODEL"),
            db_type=_env("DB_TYPE", default="sqlite", required=False),
            llm_temperature=float(_env("LLM_TEMPERATURE", default=0.7, required=False)),
            llm_timeout=int(_env("LLM_TIMEOUT", default=600, required=False)),
            firecrawl_api_key=_env("FIRECRAWL_API_KEY"),
            mongodb_session_url=_env("MONGODB_SESSION_URL", required=False),
        )


config = AppConfig.from_env()
