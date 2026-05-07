#!/user/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Dict, Union
from langchain.tools import tool, ToolRuntime

from core.utils import get_history


@tool
def tool_get_history(runtime: ToolRuntime) -> Union[dict, str]:
    """
    获取当前会话中所有经过处理的上下文，并返回列表。本工具当且仅当用户明确提出要获取历史消息/上下文时才调用
    :param runtime: 该参数对Agent/LLM自动注入，无需理会
    :return: dict，历史会话记忆，分为ai和human；或str（错误/警告信息）。
    """
    return get_history(runtime)
