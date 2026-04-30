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

import redis
from dotenv import load_dotenv

import pandas as pd
import questionary
from datetime import datetime
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from pymongo import MongoClient
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
from langchain_community.chat_message_histories import ChatMessageHistory, RedisChatMessageHistory
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool, ToolRuntime
from langchain.messages import AIMessageChunk
from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.mongodb import MongoDBSaver

import sheet_processing.tools as tools_module

warnings.filterwarnings("ignore", message="Workbook contains no default style")
dotenv_path = ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)
console = Console()

sp = """你是一个文档处理助手。请严格遵守以下规则：
1. **调用工具的前提**：只有当用户明确要求处理csv或excel文件时，才可以调用工具
2. **不调用工具的情况**：如果用户只是闲聊、询问与文件处理无关的问题，直接回答，不要调用任何工具
3. **调用前确认**：在调用工具前，必须明确知道用户想处理哪个具体文件、处理文件的什么内容
4. **不确定时反问**：如果不确定用户是否想处理文件，先反问用户确认
5. **回答完整性**：严格按照用户指令执行，不要自作主张省略、总结、修改或裁剪任何内容，尤其是工具输出必须完整原样返回
6. **工具输出处理**：调用工具后，直接将工具返回的完整内容原样呈现给用户，不得删除、裁剪、整理、总结或修改任何部分；若工具出错，重试或告知用户错误详情
记住：用户没有明确要求处理文件时，保持沉默并直接回答。"""


# 检测是否支持 rich 渲染（终端环境）
def _is_rich_available():
    """检测是否在支持 rich 的终端环境中运行"""
    return sys.stdout.isatty() and sys.stdin.isatty()


def _rich_display_message(content: str, token_usage: dict = None, total_token: int = 0):
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


def _rich_display_tool(tool_name: str, tool_args: dict, token_usage: dict = None, total_token: int = 0):
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


def _plain_display_message(content: str, token_usage: dict = None, total_token: int = 0):
    """使用普通 print 输出消息"""
    print(f"\n【Agent】{content}")
    if token_usage:
        print(f"\n\n*Token: {token_usage['total_tokens']}({token_usage['input_tokens']}/{token_usage['output_tokens']}) Total：{total_token}*")
    print()


def _plain_display_tool(tool_name: str, tool_args: dict, token_usage: dict = None, total_token: int = 0):
    """使用普通 print 输出工具调用"""
    print(f"\n【Tool: {tool_name}】")
    print(f"Args: {json.dumps(tool_args, indent=2, ensure_ascii=False)}")
    print(f"\n\n*Token: {token_usage['total_tokens']}({token_usage['input_tokens']}/{token_usage['output_tokens']}) Total：{total_token}*")


def _generate_session_id():
    """生成带日期的sessionid"""
    date_part = datetime.now().strftime("%Y%m%d")
    random_part = uuid.uuid4().hex[:8]
    return f"{date_part}-{random_part}"


def _init_llm():
    return ChatOpenAI(
        model='[DS] Deepseek V4 Flash',
        base_url='https://newapi.axuan.online/v1',
        api_key=os.getenv("API_KEY"),  # type: ignore
        temperature=0.7,
        timeout=600,
    )


def _init_checkpointer(db: str = 'mongodb'):
    if db == "mongodb":
        mongodb_client = MongoClient(str(os.getenv("MONGO_SHORTMEMORY_URL")))
        return MongoDBSaver(mongodb_client, db_name='agentbox')
    elif db == "redis":
        redis_client = RedisSaver(redis_url=str(os.getenv("REDIS_SHORTMEMORY_URL")))
        return redis_client
    else:
        raise "暂不支持其他数据库"


def _init_display():
    use_rich = _is_rich_available()
    return (
        _rich_display_message if use_rich else _plain_display_message,
        _rich_display_tool if use_rich else _plain_display_tool,
    )


def _create_agent(llm, tools, checkpointer, history_summarize):
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=sp,
        middleware=[history_summarize],
        checkpointer=checkpointer,
    )


def _process_stream_chunk(chunk, display_message, display_tool, total_token):
    latest_message = chunk["messages"][-1]
    if latest_message.type != 'ai':
        return total_token

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

    return total_token


def _run_loop(agent, config, display_message, display_tool):
    total_token = 0

    while True:
        try:
            user_input = questionary.text(">", multiline=True).ask()
        except Exception:
            user_input = input("> ")

        if user_input == 'exit':
            sys.exit()

        for chunk in agent.stream(
            input={"messages": [{"role": "user", "content": user_input}]},
            config=config,
            stream_mode="values"
        ):
            total_token = _process_stream_chunk(chunk, display_message, display_tool, total_token)


def main():
    llm = _init_llm()
    tools = [
        tools_module.tool_get_csv_excel_path,
        tools_module.tool_get_columns,
        tools_module.tool_get_columns_content,
        tools_module.tool_count_value_in_column,
        tools_module.tool_calculate_add,
        tools_module.tool_get_row_content,
        tools_module.tool_count_data_rows,
    ]
    checkpointer = _init_checkpointer()
    display_message, display_tool = _init_display()

    session_id = _generate_session_id()
    config = RunnableConfig(configurable={"thread_id": session_id})

    history_summarize = SummarizationMiddleware(
        model=llm,
        trigger=cast(tuple[Literal["tokens"], int], ("tokens", 20000)),
        keep=cast(tuple[Literal["messages"], int], ("messages", 10)),
    )

    agent = _create_agent(llm, tools, checkpointer, history_summarize)
    _run_loop(agent, config, display_message, display_tool)


if __name__ == '__main__':
    main()