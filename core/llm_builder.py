#!/user/bin/env python3
# -*- coding: utf-8 -*-

from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_openrouter import ChatOpenRouter

_PROVIDERS = {
    "openai-compatible": lambda m, b, ak, tt, to, s, eb: ChatOpenAI(model=m, base_url=b, api_key=ak, temperature=tt, timeout=to, streaming=s, extra_body=eb),
    # "openrouter": lambda m, b, ak, tt, to, r, s: ChatOpenRouter(model=m, base_url=b, api_key=ak, temperature=tt, timeout=to, reasoning_effort=r, streaming=s),
    "deepseek": lambda m, b, ak, tt, to, s, eb: ChatDeepSeek(model=m, base_url=b, api_key=ak, temperature=tt, timeout=to, streaming=s, extra_body=eb),
}


def _detect_provider(model_name: str, base_url: str):
    """根据model_name自动选择合适的llm创建器"""
    if any(sub in model_name.lower() for sub in ['deepseek', 'ds']) or 'deepseek' in base_url.lower():
        return "deepseek"
    elif 'openrouter' in base_url.lower() or model_name.lower().startswith(('openrouter/',)) or 'openrouter' in base_url.lower():
        # return "openrouter"
        return "openai-compatible"
    else:
        return "openai-compatible"


def init_llm(model_name: str, base_url: str, api_key: str, temperature: float = 0.7, timeout: int = 600, reasoning_effort=None):
    if reasoning_effort is not None and str(reasoning_effort).lower() in ['low', 'medium', 'high', 'xhigh']:
        extra_body = {
            "thinking": {"type": "enabled"},
            "reasoning_effort": str(reasoning_effort).lower(),
        }
    else:
        extra_body = {
            "thinking": {"type": "disabled"},
        }
    provider = _detect_provider(model_name, base_url)
    builder = _PROVIDERS[provider]
    return builder(model_name, base_url, api_key, temperature, timeout, True, extra_body)
