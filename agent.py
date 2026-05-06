"""
读取所有指定目录下的csv路径->
对每个csv获取表头->
获取指定平台的行数
"""

import os
import uuid
import sys
import re
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


def _init_display():
    use_rich = _is_rich_available()
    return (
        _rich_display_message if use_rich else _plain_display_message,
        _rich_display_tool if use_rich else _plain_display_tool,
    )


class AgentBox:
    def __init__(self, db: str = 'mongodb'):
        self.llm = ChatOpenAI(
            model=str(os.getenv("CHAT_MODEL")),
            base_url=str(os.getenv("BASE_URL")),
            api_key=os.getenv("API_KEY"),  # type: ignore
            temperature=0.7,
            timeout=600,
        )

        self.tools = [
            tools_module.tool_get_csv_excel_path,
            tools_module.tool_get_columns,
            tools_module.tool_get_columns_content,
            tools_module.tool_count_value_in_column,
            tools_module.tool_calculate_add,
            tools_module.tool_get_row_content,
            tools_module.tool_count_data_rows,
        ]

        self.checkpointer = self._init_checkpointer(db)
        self.display_message, self.display_tool = _init_display()
        self.session_id = self._generate_session_id()
        self.config = RunnableConfig(configurable={"thread_id": self.session_id})

        self.history_summarize = SummarizationMiddleware(
            model=self.llm,
            trigger=cast(tuple[Literal["tokens"], int], ("tokens", 20000)),
            keep=cast(tuple[Literal["messages"], int], ("messages", 10)),
        )

        self.agent = self._build_agent()

        self.total_token = 0

        # db
        self.mongodb_collection = self._init_db(db='mongodb')  # 会话记忆数据库

    def _build_agent(self):
        return create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=sp,
            # middleware=[self.history_summarize],
            checkpointer=self.checkpointer,
        )

    def _generate_session_id(self):
        """生成带日期的sessionid"""
        date_part = datetime.now().strftime("%Y%m%d")
        random_part = uuid.uuid4().hex[:8]
        return f"{date_part}-{random_part}"

    def _check_session_id(session_id: str):
        """检查是否为正确的sessionid"""
        pattern = r'^\d{8}-[0-9a-f]{8}$'
        return bool(re.fullmatch(pattern, session_id))

    def _init_checkpointer(self, db: str):
        if db == "mongodb":
            mongodb_client = MongoClient(str(os.getenv("MONGO_SHORTMEMORY_URL")))
            return MongoDBSaver(mongodb_client, db_name='agentbox')
        # elif db == "redis":
        #     return RedisSaver(redis_url=str(os.getenv("REDIS_SHORTMEMORY_URL")))
        else:
            raise ValueError("暂不支持其他数据库")

    def _init_db(self, db: str = 'mongodb'):
        if db == "mongodb":
            mongodb_client = MongoClient(str(os.getenv("MONGO_SHORTMEMORY_URL")))
            _db = mongodb_client["agentbox"]
            return _db["checkpoints"]
        raise ValueError("暂不支持其他数据库")

    def _get_session_ids(self):
        """从会话记忆数据库中获取所有符合条件的sessionid"""
        cursor = self.mongodb_collection.find({}, {"thread_id": 1, "_id": 0})
        pattern = re.compile(r'^\d{8}-')
        return list(set([
            doc["thread_id"]
            for doc in cursor
            if "thread_id" in doc and pattern.match(doc["thread_id"])
        ]))

    def _check_session_id_available(self, session_id: str):
        """检查该sessionid是否存在于数据库中"""
        session_ids = self._get_session_ids()
        if session_id in session_ids:
            return True
        return False

    def _select_session(self):
        session_ids = self._get_session_ids()
        if not session_ids:
            console.print(Panel("没有找到历史会话", title="📋 会话列表", border_style="yellow"))
            return None

        groups: Dict[str, list] = {}
        for sid in sorted(session_ids, reverse=True):
            date_part = sid[:8]
            groups.setdefault(date_part, []).append(sid)

        choices = []
        for date_part, ids in groups.items():
            formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
            choices.append(questionary.Separator(f"── {formatted_date} ──"))
            for sid in ids:
                choices.append(questionary.Choice(title=f"  {sid}", value=sid))

        return questionary.select(
            "选择要恢复的会话（↑↓ 选择，回车确认，Esc 取消）",
            choices=choices,
            use_indicator=True,
        ).ask()

    def _process_stream_chunk(self, chunk):
        latest_message = chunk["messages"][-1]
        if latest_message.type != 'ai':
            return

        token_usage = None
        if latest_message.usage_metadata:
            token_usage = latest_message.usage_metadata
            if token_usage:
                self.total_token += token_usage['total_tokens']

        if latest_message.content:
            self.display_message(latest_message.content, token_usage, self.total_token)

        if latest_message.tool_calls:
            for tc in latest_message.tool_calls:
                self.display_tool(tc['name'], tc['args'], token_usage, self.total_token)

    def run(self):
        while True:
            try:
                user_input = questionary.text(">", multiline=True).ask()
            except Exception:
                user_input = input("> ")

            if user_input in ['exit', 'exit\n', 'quit', 'quit\n']:
                console.print(Panel(
                    f"本次会话已保存，可通过/session或/session session_id来恢复会话记忆",
                    title=f"会话保存 {self.config['configurable']['thread_id']}",
                    border_style="dark_red",
                ))
                sys.exit()

            elif re.search(r"/session( [\w-]+)?$", user_input):
                if user_input == "/session":
                    selected = self._select_session()
                else:
                    selected = user_input.split(" ")[-1]
                if not self._check_session_id(str(selected)):
                    console.print("[bold red]输入的session_id格式错误[/bold red]，正确格式为 [green]yyyymmdd-12345678[/green]")
                    continue
                if selected:
                    if not self._check_session_id_available(str(selected)):
                        console.print("[bold red]该session_id不存在，请检查后重试[/bold red]")
                        continue
                    self.config["configurable"]["thread_id"] = selected
                    self.agent = self._build_agent()  # 重新构建agent以读取历史记录
                    console.print(Panel(
                        f"已切换到会话: {selected}",
                        title="✅ 会话切换",
                        border_style="green",
                    ))
                continue

            for chunk in self.agent.stream(
                input={"messages": [{"role": "user", "content": user_input}]},
                config=self.config,
                stream_mode="values",
            ):
                self._process_stream_chunk(chunk)


def main():
    agent_box = AgentBox()
    agent_box.run()


if __name__ == '__main__':
    main()
