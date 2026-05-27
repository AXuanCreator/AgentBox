#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
from urllib.parse import urlparse

from firecrawl import Firecrawl

from core.schemas import ResponseCode, ToolResponse

dotenv_path = ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)


def fetch_single_url_to_md(url: str) -> dict:
    check_url = urlparse(url)
    if not check_url.scheme or not check_url.netloc:
        return ToolResponse(success=False, code=ResponseCode.URL_ERROR, message="URL格式错误", data=None).model_dump()
    try:
        firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
        scrape_result = firecrawl.scrape(url, formats=['markdown'], remove_base64_images=True, only_main_content=True)
    except Exception as e:
        return ToolResponse(success=False, code=ResponseCode.GENERIC_ERROR, message=f"错误：{e}", data=None).model_dump()

    return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"获取单个URL页面内容成功", data=scrape_result.markdown).model_dump()


def search_online_by_query(query: str, limit: int = 5) -> dict:
    try:
        firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
        results_raw = firecrawl.search(
            query=query,
            limit=limit
        )
        results = results_raw.web
    except Exception as e:
        return ToolResponse(success=False, code=ResponseCode.GENERIC_ERROR, message=f"错误：{e}", data=None).model_dump()

    if len(results) == 0:
        return ToolResponse(success=False, code=ResponseCode.GENERIC_ERROR, message=f"无搜索结果", data=None).model_dump()

    results_dict = {f'result_{idx}': {"url": item.url, "title": item.title, "description": item.description} for idx, item in enumerate(results)}
    return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"获取网络搜索内容成功", data=results_dict).model_dump()


if __name__ == "__main__":
    print()
