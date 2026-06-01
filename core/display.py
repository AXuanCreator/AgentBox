#!/user/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel


class RichRenderer:
    def __init__(self):
        self.console = Console()

    def display_message(self, content: str, token_usage: dict = None, total_token: int = 0):
        """使用 rich 渲染Agent输出"""
        display_content = content
        if token_usage:
            display_content += f"\n\n*Token: {token_usage['total_tokens']}({token_usage['input_tokens']}/{token_usage['output_tokens']}) Total：{total_token}*"

        self.console.print(Panel(
            Markdown(display_content),
            title="🤖 Agent",
            border_style="cyan",
            expand=False,
            padding=(0, 1)
        ))

    def display_tool(self, tool_name: str, tool_args: dict, token_usage: dict = None, total_token: int = 0):
        """使用 rich 渲染工具调用"""
        display_content = f"**Args**:{json.dumps(tool_args, indent=2, ensure_ascii=False)}"
        display_content += f"\n\n*Token: {token_usage['total_tokens']}({token_usage['input_tokens']}/{token_usage['output_tokens']}) Total：{total_token}*"
        self.console.print(Panel(
            Markdown(display_content),
            title=f"🔧 Tool: {tool_name}",
            border_style="yellow",
            expand=False,
            padding=(0, 1)
        ))

    def print(self, content: str):
        self.console.print(content)

    def print_panel(self, content: str, title: str = "", border_style: str = ""):
        self.console.print(Panel(content, title=title, border_style=border_style, expand=False, padding=(0, 1)))


class PlainRenderer:
    @staticmethod
    def display_message(content: str, token_usage: dict = None, total_token: int = 0):
        """使用普通 print 输出消息"""
        print(f"\n【Agent】{content}")
        if token_usage:
            print(f"\n\n*Token: {token_usage['total_tokens']}({token_usage['input_tokens']}/{token_usage['output_tokens']}) Total：{total_token}*")
        print()

    @staticmethod
    def display_tool(tool_name: str, tool_args: dict, token_usage: dict = None, total_token: int = 0):
        """使用普通 print 输出工具调用"""
        print(f"\n【Tool: {tool_name}】")
        print(f"Args: {json.dumps(tool_args, indent=2, ensure_ascii=False)}")
        print(f"\n\n*Token: {token_usage['total_tokens']}({token_usage['input_tokens']}/{token_usage['output_tokens']}) Total：{total_token}*")

    @staticmethod
    def print(content: str):
        clean = re.sub(r'\[/?\w+[^\]]*\]', '', content)
        print(clean)

    @staticmethod
    def print_panel(content: str, title: str = "", border_style: str = ""):
        print(f"\n── {title} ──")
        print(content)
        print()


def init_display_renderer(debug=False):
    if debug:
        return PlainRenderer()
    return RichRenderer() if sys.stdout.isatty() and sys.stdin.isatty() else PlainRenderer()


def process_stream_chunk(chunk, renderer, total_token: int):
    """处理agent流式处理时返回的块"""
    latest_message = chunk["messages"][-1]
    if latest_message.type != 'ai':
        return total_token

    token_usage = None
    if latest_message.usage_metadata:
        token_usage = latest_message.usage_metadata
        if token_usage:
            total_token += token_usage['total_tokens']

    if latest_message.content:
        renderer.display_message(latest_message.content, token_usage, total_token)

    if latest_message.tool_calls:
        for tc in latest_message.tool_calls:
            renderer.display_tool(tc['name'], tc['args'], token_usage, total_token)

    return total_token
