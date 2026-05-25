#!/user/bin/env python3
# -*- coding: utf-8 -*-

from langchain.tools import tool, ToolRuntime

from core import fetch_utils


@tool
def tool_fetch_single_url_to_md(url: str) -> dict:
    """
    根据指定的 URL 获取该网页的内容，只获取这一页，不获取它的子页面内容。
    :param url: 指定网址
    :return:  dict：结果或错误信息
    """
    return fetch_utils.fetch_single_url_to_md(url)
