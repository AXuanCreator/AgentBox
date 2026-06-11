#!/user/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import sys
import contextlib

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

class RichRenderer:
    def __init__(self):
        self.console = Console()

    def display_message(self, content_blocks, token_usage: dict = None, total_token: int = 0):
        """使用 rich 渲染Agent输出"""
        display_content = ""
        renderables = []
        for cb in content_blocks:
            if cb['type'] == 'reasoning':
                # reasoning_panel = Panel(
                #     Text(cb['reasoning'], style="gray23"),
                #     title="Thinking",
                #     border_style="gray23",
                #     expand=True,
                #     padding=(0, 1),
                # )
                renderables.append(Text(f"Thinking: {cb['reasoning']}\n", style="gray23"))
            elif cb['type'] == 'text':
                renderables.append(Markdown(cb['text']))

        if token_usage:
            renderables.append(Text(f"Token: {token_usage['total_tokens']}({token_usage['input_tokens']}/{token_usage['output_tokens']}) Total：{total_token}", style="gray42"))

        self.console.print(Panel(
            Group(*renderables),
            title="Agent",
            border_style="cyan",
            expand=True,
            padding=(0, 1)
        ))

    def display_tool(self, tool_name: str, tool_args: dict, token_usage: dict = None, total_token: int = 0):
        """使用 rich 渲染工具调用"""
        display_content = f"[grey35]Args:{json.dumps(tool_args, indent=2, ensure_ascii=False)}"
        display_content += f"\n\nToken: {token_usage['total_tokens']}({token_usage['input_tokens']}/{token_usage['output_tokens']}) Total：{total_token}[/grey35]"
        self.console.print(Panel(
            display_content,
            title=f"Tool: {tool_name}",
            border_style="grey35",
            expand=True,
            padding=(0, 1)
        ))

    def print(self, content: str):
        self.console.print(content)

    def print_panel(self, content: str, title: str = "", border_style: str = ""):
        self.console.print(Panel(content, title=title, border_style=border_style, expand=False, padding=(0, 1)))

    def status(self, msg: str = "正在处理中..."):
        return self.console.status(msg, spinner='flip')


class PlainRenderer:
    @staticmethod
    def display_message(content_blocks, token_usage: dict = None, total_token: int = 0):
        """使用普通 print 输出消息"""
        print(f"\n【Agent】{content_blocks}")
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

    @staticmethod
    def status(msg: str = "正在处理中..."):
        return contextlib.nullcontext()


def init_display_renderer(debug=False):
    if debug:
        return PlainRenderer()
    return RichRenderer() if sys.stdout.isatty() and sys.stdin.isatty() else PlainRenderer()


def process_stream_chunk(chunk, renderer, total_token: int, status_obj=None):
    """处理agent流式处理时返回的块"""
    latest_message = chunk["messages"][-1]
    if latest_message.type != 'ai':
        return total_token

    token_usage = None
    if latest_message.usage_metadata:
        token_usage = latest_message.usage_metadata
        if token_usage:
            total_token += token_usage['total_tokens']

    if status_obj is not None:
        status_obj.update("Thinking...")

    if latest_message.content_blocks:
        renderer.display_message(latest_message.content_blocks, token_usage, total_token)

    if latest_message.tool_calls:
        if status_obj is not None:
            status_obj.update("Running Tools...")
        for tc in latest_message.tool_calls:
            renderer.display_tool(tc['name'], tc['args'], token_usage, total_token)

    return total_token
