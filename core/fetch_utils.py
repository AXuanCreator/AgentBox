#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

from urllib.parse import urlparse
from firecrawl import Firecrawl

dotenv_path = ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)


def fetch_single_url_to_md(url: str) -> str:
    check_url = urlparse(url)
    if not check_url.scheme or not check_url.netloc:
        return "错误：URL无效或格式不正确，请确认"
    try:
        firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))
        scrape_result = firecrawl.scrape(url, formats=['markdown'], remove_base64_images=True, only_main_content=True)
    except Exception as e:
        return "错误：URL解析错误"

    return scrape_result.markdown
