"""
读取所有指定目录下的csv路径->
对每个csv获取表头->
获取指定平台的行数
"""

import os
import uuid
import sys
import inspect
import json
import warnings
from typing import List, Dict, Union, cast, Literal
from dotenv import load_dotenv

import pandas as pd
import questionary
from langchain.agents.structured_output import ToolStrategy
from pandas import DataFrame
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.console import Group
from pathlib import Path

from langchain.agents.middleware import SummarizationMiddleware
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain.agents import create_agent
from langchain_classic.agents import AgentExecutor
from langchain_classic.schema.runnable import configurable
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessageChunk, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_community.chat_message_histories import ChatMessageHistory
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool, ToolRuntime
from langchain.messages import AIMessageChunk

import sheet_processing.tools as tools_module

warnings.filterwarnings("ignore", message="Workbook contains no default style")
dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)


sp = """你是一个文档处理助手。请严格遵守以下规则：
1. **调用工具的前提**：只有当用户明确要求处理csv或excel文件时，才可以调用工具
2. **不调用工具的情况**：如果用户只是闲聊、询问与文件处理无关的问题，直接回答，不要调用任何工具
3. **调用前确认**：在调用工具前，必须明确知道用户想处理哪个具体文件、处理文件的什么内容
4. **不确定时反问**：如果不确定用户是否想处理文件，先反问用户确认
5. **回答完整性**：严格按照用户指令执行，不要自作主张省略、总结、修改或裁剪任何内容，尤其是工具输出必须完整原样返回
6. **工具输出处理**：调用工具后，直接将工具返回的完整内容原样呈现给用户，不得删除、裁剪、整理、总结或修改任何部分；若工具出错，重试或告知用户错误详情
记住：用户没有明确要求处理文件时，保持沉默并直接回答。"""



# 检测是否支持 rich 渲染（终端环境）
def is_rich_available():
    """检测是否在支持 rich 的终端环境中运行"""
    return sys.stdout.isatty() and sys.stdin.isatty()


def rich_display_message(content: str, token_usage: dict = None, total_token: int = 0):
    """使用 rich 渲染消息"""
    display_content = content
    if token_usage:
        display_content += f"\n\n*Token: {token_usage['total_tokens']}({token_usage['input_tokens']}/{token_usage['output_tokens']}) Total：{total_token}*"

    console.print(Panel(
        Markdown(display_content),
        title="🤖 Agent",
        border_style="cyan",
        expand=False,
        padding=(1, 2)
    ))


def rich_display_tool(tool_name: str, tool_args: dict, token_usage: dict = None, total_token: int = 0):
    """使用 rich 渲染工具调用"""
    display_content = f"**Args**:\n```json\n{json.dumps(tool_args, indent=2, ensure_ascii=False)}\n```"
    display_content += f"\n\n*Token: {token_usage['total_tokens']}({token_usage['input_tokens']}/{token_usage['output_tokens']}) Total：{total_token}*"
    console.print(Panel(
        Markdown(display_content),
        title=f"🔧 Tool: {tool_name}",
        border_style="yellow",
        expand=False,
        padding=(1, 2)
    ))


def plain_display_message(content: str, token_usage: dict = None, total_token: int = 0):
    """使用普通 print 输出消息"""
    print(f"\n【Agent】{content}")
    if token_usage:
        print(f"\n\n*Token: {token_usage['total_tokens']}({token_usage['input_tokens']}/{token_usage['output_tokens']}) Total：{total_token}*")
    print()


def plain_display_tool(tool_name: str, tool_args: dict, token_usage: dict = None, total_token: int = 0):
    """使用普通 print 输出工具调用"""
    print(f"\n【Tool: {tool_name}】")
    print(f"Args: {json.dumps(tool_args, indent=2, ensure_ascii=False)}")
    print(f"\n\n*Token: {token_usage['total_tokens']}({token_usage['input_tokens']}/{token_usage['output_tokens']}) Total：{total_token}*")


llm = ChatOpenAI(
    # model='[OR] Ling 2.6 Flash',
    # model = '[OR] HY 3 Free',
    model = '[VE] Doubao 2 Lite T',
    base_url='https://newapi.axuan.online/v1',
    api_key=os.getenv("API_KEY"),  # type: ignore
    temperature=0.7,
    timeout=600,  # 600s超时
)

tools = [tools_module.tool_get_csv_excel_path, tools_module.tool_get_columns, tools_module.tool_get_columns_content, tools_module.tool_count_value_in_column,
         tools_module.tool_calculate_add, tools_module.tool_get_row_content, tools_module.tool_count_data_rows]

config = RunnableConfig(
    configurable={
        "thread_id": 1
    }
)

history_summarize = SummarizationMiddleware(
    model=llm,
    trigger=cast(tuple[Literal["tokens"], int], ("tokens", 4000)),  # 历史消息超过4000tokens触发总结
    keep=cast(tuple[Literal["messages"], int], ("messages", 20))  # 保留最近20条消息不动，其余历史消息均总结
)

agent = create_agent(model=llm,
                     tools=tools,
                     system_prompt=sp,
                     middleware=[history_summarize],
                     checkpointer=InMemorySaver(),
                     # response_format=ToolStrategy
                     )

console = Console()

# 根据环境选择输出函数
use_rich = is_rich_available()

display_message = rich_display_message if use_rich else plain_display_message
display_tool = rich_display_tool if use_rich else plain_display_tool

total_token = 0  # 总Token使用数
# 主循环
while True:
    try:
        user_input = questionary.text(">", multiline=True).ask()
    except Exception:
        user_input = input("> ")

    if user_input == 'exit':
        sys.exit()


    for chunk in agent.stream(input={
        "messages": [{"role": "user", "content": user_input}]}, config=config, stream_mode="values"):
        latest_message = chunk["messages"][-1]
        if latest_message.type == 'ai':
            # 获取当前AIMessage的token用量
            token_usage = None
            if latest_message.usage_metadata:
                token_usage = latest_message.usage_metadata
                if token_usage:
                    total_token += token_usage['total_tokens']

            if latest_message.content:
                display_message(latest_message.content, token_usage, total_token)

            if latest_message.tool_calls:
                for tc in latest_message.tool_calls:
                    display_tool(tc['name'], tc['args'], token_usage, total_token)

    # buffer = ""

    # reasoning_buffer = ""          # 推理过程缓存
    # text_buffer = ""               # 最终回复文本缓存
    # active_tool_calls = {}        # 正在构建的工具调用：id -> {"name": "", "args": ""}
    # token_usage_text = ""         # Token 统计文本
    #
    # with Live(Panel(Markdown(buffer), title="🤖 Agent", border_style="blue", padding=(1, 2)), refresh_per_second=10) as live:
    #     for token, metadata in agent.stream(input={
    #         "messages": [{"role": "user", "content": user_input}]}, config=config, stream_mode="messages"):
    #         if isinstance(token, AIMessageChunk):
    #             # 1. 处理推理块（content_blocks 中的 reasoning）
    #             if hasattr(token, 'content_blocks'):
    #                 reasoning = [b for b in token.content_blocks if b["type"] == "reasoning"]
    #                 text_blocks = [b for b in token.content_blocks if b["type"] == "text"]
    #             else:
    #                 reasoning = []
    #                 text_blocks = []
    #             if reasoning:
    #                 reasoning_buffer += f"*[thinking] {reasoning[0]['reasoning']}*\n"
    #             if text_blocks:
    #                 text_buffer += text_blocks[0]["text"]
    #             # 如果没有 content_blocks，退回到 token.content（向后兼容）
    #             elif token.content and isinstance(token.content, str):
    #                 text_buffer += token.content
    #
    #             # 2. 处理流式工具调用（tool_call_chunks）
    #             if hasattr(token, 'tool_call_chunks') and token.tool_call_chunks:
    #                 for tc in token.tool_call_chunks:
    #                     tc_id = tc.get('id')
    #                     if not tc_id:
    #                         continue
    #                     if tc_id not in active_tool_calls:
    #                         active_tool_calls[tc_id] = {"name": "", "args": ""}
    #                     if tc.get('name'):
    #                         active_tool_calls[tc_id]['name'] = tc['name']
    #                     if tc.get('args'):
    #                         active_tool_calls[tc_id]['args'] += tc['args']
    #
    #             # 3. 提取 Token 用量（通常出现在最后一个 chunk）
    #             if hasattr(token, 'usage_metadata') and token.usage_metadata:
    #                 u = token.usage_metadata
    #                 if isinstance(u, dict) and 'total_tokens' in u:
    #                     token_usage_text = f"\n\n💻 **Token Usage:** {u['total_tokens']} (in: {u['input_tokens']}, out: {u['output_tokens']})"
    #
    #         elif isinstance(token, ToolMessage):
    #             # 4. 工具执行完成，将对应工具调用面板插入文本缓冲
    #             tc_id = token.tool_call_id
    #             if tc_id in active_tool_calls:
    #                 tool = active_tool_calls.pop(tc_id)
    #                 # 尝试格式化 args 为美观 JSON
    #                 try:
    #                     args_obj = json.loads(tool['args']) if tool['args'] else {}
    #                     args_str = json.dumps(args_obj, indent=2, ensure_ascii=False)
    #                 except:
    #                     args_str = tool['args']
    #                 tool_panel = f"🔧 **Tool Call: {tool['name']}**\n```json\n{args_str}\n```"
    #                 text_buffer += f"\n\n{tool_panel}\n"
    #
    #         # 5. 更新实时显示面板
    #         display_content = ""
    #         if reasoning_buffer:
    #             display_content += reasoning_buffer
    #         if text_buffer:
    #             display_content += text_buffer
    #         if active_tool_calls:
    #             in_progress = "\n".join(
    #                 [f"⚙️ Calling `{info['name']}`: `{info['args'][:60]}`..." for info in active_tool_calls.values()]
    #             )
    #             display_content += f"\n\n{in_progress}"
    #         if token_usage_text:
    #             display_content += token_usage_text
    #
    #         # 用 Markdown 渲染，使代码块、粗体等生效
    #         live.update(Panel(Markdown(display_content), title="🤖 Agent", border_style="blue", padding=(1, 2)))
