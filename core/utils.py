#!/user/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Dict, Union
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage


def get_history(runtime: ToolRuntime) -> Union[dict, str]:
    try:
        history_messages = runtime.state["messages"]
    except Exception as e:
        return f"错误：获取历史消息失败，详细错误：{e}"
    history = {
        f"{f'Human_{idx}' if isinstance(m, HumanMessage) else f'AI_{idx}'}": m.content
        for idx, m in enumerate(history_messages)
    }

    return history
