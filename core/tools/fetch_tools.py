#!/user/bin/env python3
# -*- coding: utf-8 -*-

from langchain.tools import tool

from core.impl import fetch_impl


@tool
def tool_fetch_single_url_to_md(url: str) -> dict:
    """
    根据指定的 URL 获取该网页的内容，只获取这一页，不获取它的子页面内容。
    :param url: 指定网址
    :return: dict：结果或错误信息
    """
    return fetch_impl.fetch_single_url_to_md(url)


@tool
def tool_search_online_by_query(query: str, limit: int = 10) -> dict:
    """
    通过联网搜索引擎获取与查询内容相关的网页信息，包含url、title、description信息。
    :param query: 搜索词，可以是关键词或完整问句
    :param limit: 期望返回的网页结果数量
    :return: dict：结果或错误信息
    """
    return fetch_impl.search_online_by_query(query, limit)
