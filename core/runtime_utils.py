#!/user/bin/env python3
# -*- coding: utf-8 -*-
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage

from core.schemas import ResponseCode, ToolResponse


def get_history(runtime: ToolRuntime) -> dict:
    try:
        history_messages = runtime.state["messages"]
    except Exception as e:
        return ToolResponse(success=False, code=ResponseCode.HISTORY_ERROR, message=str(e), data=None).model_dump()

    history = {
        f"{f'Human_{idx}' if isinstance(m, HumanMessage) else f'AI_{idx}'}": m.content
        for idx, m in enumerate(history_messages)
    }
    return ToolResponse(success=True, code=ResponseCode.SUCCESS, message='获取历史成功', data=history).model_dump()