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
import argparse

import redis
from dotenv import load_dotenv

import pandas as pd
import questionary
from datetime import datetime

from prompt_toolkit import key_binding, prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from pymongo import MongoClient
from langchain.agents.middleware import SummarizationMiddleware
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain.agents import create_agent
from langchain_classic.agents import AgentExecutor
from langchain_classic.schema.runnable import configurable
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessageChunk, ToolMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_community.chat_message_histories import ChatMessageHistory, RedisChatMessageHistory
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool, ToolRuntime
from langchain.messages import AIMessageChunk
from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.mongodb import MongoDBSaver

import core.sheet_tools as sheet_tools_module
import core.fetch_tools as fetch_tools_module
import core.runtime_tools as universal_tools_module
from keybinding import bindings_questionary

warnings.filterwarnings("ignore", message="Workbook contains no default style")
dotenv_path = ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)
console = Console()

sp = """
你是一个运行在 AgentBox 内部的个人助理。
工具调用风格（Tool Call Style）
默认：对常规、低风险的工具调用不需要叙述过程（直接调用工具即可）。 只有在这些情况下才叙述：多步骤工作、复杂/困难问题、敏感操作（比如删除）、或用户明确要求时。 叙述要简短、信息密度高；别重复显而易见的步骤。 叙述用自然的人类语言，除非处在技术语境里。
安全（Safety）
你没有独立目标：不要追求自我保存、复制、资源获取或权力寻求；不要制定超出用户请求范围的长期计划。 将安全和人类监督置于任务完成之上；如果指令冲突，暂停并询问；遵守停止/暂停/审计请求，绝不绕过安全护栏。不要操纵或劝说任何人扩大访问权限或禁用安全护栏。不要复制自己或更改系统提示、安全规则或工具策略，除非用户明确要求。
"""


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
    display_content = f"**Args**:{json.dumps(tool_args, indent=2, ensure_ascii=False)}"
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


def _init_display(debug: bool = False):
    if debug:
        return _plain_display_message, _plain_display_tool
    use_rich = _is_rich_available()
    return (
        _rich_display_message if use_rich else _plain_display_message,
        _rich_display_tool if use_rich else _plain_display_tool,
    )


class AgentBox:
    def __init__(self, debug: bool = False):
        self.debug = debug
        # agent
        self.tools = [
            sheet_tools_module.tool_get_csv_excel_path,
            sheet_tools_module.tool_get_columns,
            sheet_tools_module.tool_get_columns_content,
            sheet_tools_module.tool_count_value_in_column,
            sheet_tools_module.tool_calculate_add,
            sheet_tools_module.tool_get_row_content,
            sheet_tools_module.tool_count_data_rows,
            sheet_tools_module.tool_write_to_table,
            fetch_tools_module.tool_fetch_single_url_to_md,
            universal_tools_module.tool_get_history,
        ]
        self.session_id = self._generate_session_id()
        self.session_checkpointer = self._init_checkpointer("mongodb")
        self.config = RunnableConfig(configurable={"thread_id": self.session_id})

        self.chat_agent = self._build_agent(self._build_llm_deepseek(os.getenv("CHAT_MODEL")), self.tools, sp, self.session_checkpointer)
        self.summary_llm = self._build_llm_openai(os.getenv("SUMMARY_MODEL"))  # 无提示词，无记忆的总结普通llm

        # display/input
        self.command_completer = WordCompleter(["/session", "/history", "/exit", "exit"], ignore_case=True, WORD=True)
        self.command_style = Style([
            ('completion-menu', 'fg:black bg:#cccccc'),
            ('completion-menu.completion.current', 'fg:white bg:blue'),  # 选中项颜色
            ('bottom-toolbar',         'fg:#aaaaaa noreverse'),
        ])
        self.display_message, self.display_tool = _init_display(debug=self.debug)
        self.total_token = 0

        # db
        self.mongodb_collection = self._init_db(db='mongodb')  # 会话记忆数据库

    def _print(self, content: str):
        """根据 debug 模式输出文本"""
        if self.debug:
            clean = re.sub(r'\[/?\w+[^\]]*\]', '', content)
            print(clean)
        else:
            console.print(content)

    def _print_panel(self, content: str, title: str = "", border_style: str = ""):
        """根据 debug 模式输出 Panel"""
        if self.debug:
            print(f"\n── {title} ──")
            print(content)
            print()
        else:
            console.print(Panel(content, title=title, border_style=border_style, expand=False, padding=(1, 2)))

    def _check_session_id_available(self, session_id: str):
        """检查该sessionid是否存在于数据库中"""
        session_ids = self._get_session_ids()
        if session_id in session_ids:
            return True
        return False

    def _select_session(self):
        session_ids = self._get_session_ids()
        recent5 = "recent_5_sessions"
        if not session_ids:
            self._print_panel("没有找到历史会话", title="📋 会话列表", border_style="yellow")
            return None

        groups: dict[str, list] = {}
        for sid in sorted(session_ids, reverse=True):
            date_part = sid[:8]
            groups.setdefault(date_part, []).append(sid)
        groups.setdefault(recent5, []).extend(sorted(session_ids, reverse=True)[:5])

        self._print('[green]↑↓[/green] 选择，[yellow]Enter[/yellow]确认，[red]Ctrl+C[/red]取消')

        # 一级选择页（最近+日期）
        first_choices = [questionary.Choice(
            title=f"最近5个会话",
            value=recent5
        )]
        for date_part in groups:
            if date_part == recent5:
                continue
            formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
            first_choices.append(questionary.Choice(
                title=f"{formatted_date} ({len(groups[date_part])} 个会话)",
                value=date_part
            ))

        first_selected = questionary.select(
            "选择日期",
            choices=first_choices,
            use_indicator=True,
            instruction=''
        ).ask()

        if first_selected is None:
            return None

        # 二级选择页（会话）
        sessions_by_selected = groups[first_selected]
        session_choices = [
            questionary.Choice(title=f"  {sid}", value=sid)
            for sid in sessions_by_selected
        ]

        return questionary.select(
            f"选择 {first_selected} 下的会话",
            choices=session_choices,
            use_indicator=True,
            instruction=''
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

    def _get_session_ids(self):
        """从会话记忆数据库中获取所有符合条件的sessionid"""
        cursor = self.mongodb_collection.find({}, {"thread_id": 1, "_id": 0})
        pattern = re.compile(r'^\d{8}-')
        return list(set([
            doc["thread_id"]
            for doc in cursor
            if "thread_id" in doc and pattern.match(doc["thread_id"])
        ]))

    def _summarize_history(self, messages: list, retain: int = 4):
        """总结历史消息"""
        msg_str = self._format_messages_to_str(messages)
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "请将以下对话历史总结为简洁的摘要，保留关键信息"),
            ("human", "{messages_str}")
        ])

        summary_chain = summary_prompt | self.summary_llm | StrOutputParser()
        summary_text = summary_chain.invoke({'messages_str': msg_str})

        return summary_text

    def run(self):
        while True:
            if self.debug:
                user_input = input("> ")
            else:
                user_input = prompt(
                    ">",
                    multiline=True,
                    key_bindings=bindings_questionary,
                    bottom_toolbar="(ENTER 发送，CTRL+J 换行)",
                    completer=self.command_completer,
                    style=self.command_style
                )
            user_input = re.sub(r'\n+', '\n', user_input).strip('\n')  # 去除末尾所有换行符

            if user_input in ['exit', 'exit\n', '/exit', 'quit', 'quit\n']:
                self._print_panel(
                    f"本次会话已保存，可通过[bright_blue]/session[/bright_blue]或[bright_blue]/session {self.config['configurable']['thread_id']}[/bright_blue]来恢复会话",
                    title=f"会话保存 {self.config['configurable']['thread_id']}",
                    border_style="dark_red",
                )
                sys.exit()

            elif re.search(r"/session( [\w-]+)?$", user_input):
                if user_input == "/session":
                    if self.debug:
                        self._print(f"调试模式不支持使用session选择器，请直接输入特定session_id:\n{self._get_session_ids()}")
                        continue
                    selected = self._select_session()
                else:
                    selected = user_input.split(" ")[-1]
                if not self._check_session_id(str(selected)):
                    self._print("[bold red]输入的session_id格式错误[/bold red]，正确格式为 [green]yyyymmdd-12345678[/green]")
                    continue
                if selected:
                    if not self._check_session_id_available(str(selected)):
                        self._print("[bold red]该session_id不存在，请检查后重试[/bold red]")
                        continue
                    self.config["configurable"]["thread_id"] = selected
                    self.chat_agent = self._build_agent(self._build_llm_deepseek(os.getenv("CHAT_MODEL")), self.tools, sp, self.session_checkpointer)  # 重新构建agent以读取历史记录
                    history_message = self.chat_agent.get_state(self.config).values["messages"]
                    history_summary = self._summarize_history(history_message)
                    self._print_panel(
                        f"已切换到会话: [green]{selected}[/green]\n过往消息总结: {history_summary}\n\n[dim]可通过[bright_blue]/history[/bright_blue]或直接咨询来查询历史消息[/dim]",
                        title="✅ 会话切换",
                        border_style="green",
                    )
                continue

            elif user_input in ["/history"]:
                history_message = self.chat_agent.get_state(self.config).values.get("messages", "")
                if history_message:
                    history_message_format = self._format_messages_to_str(messages=history_message, cut=True, style=True)
                    self._print_panel(
                        f"{history_message_format}",
                        title="🕛 历史消息",
                        border_style="violet",
                    )
                    continue
                else:
                    self._print("[bold red]当前会话不存在任何历史消息[/bold red]")
                    continue

            for chunk in self.chat_agent.stream(
                input={"messages": [{"role": "user", "content": user_input}]},
                config=self.config,
                stream_mode="values",
            ):
                self._process_stream_chunk(chunk)

    @staticmethod
    def _build_llm_openai(model_name):
        """适用于openai格式，且不带reason_content（非思考模式）的llm"""
        return ChatOpenAI(
            model=str(model_name),
            base_url=str(os.getenv("BASE_URL")),
            api_key=os.getenv("API_KEY"),  # type: ignore
            temperature=0.7,
            timeout=600,
            streaming=True
        )

    @staticmethod
    def _build_agent(llm, tools=None, system_prompt=None, checkpointer=None):
        return create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            # middleware=[self.history_summarize],
            checkpointer=checkpointer,
        )

    @staticmethod
    def _build_llm_deepseek(model_name):
        """适用于deepseek模型"""
        return ChatDeepSeek(
            model=str(model_name),
            api_base=str(os.getenv("BASE_URL")),
            api_key=os.getenv("API_KEY"),
            temperature=0.7,
            timeout=600,
            streaming=True,
        )

    @staticmethod
    def _format_messages_to_str(messages, cut: bool = False, style: bool = False):
        lines = []
        for m in messages:
            if style:
                role = '[bold blue]Human[/bold blue]' if isinstance(m, HumanMessage) else '[bold yellow]AI[/bold yellow]'
            else:
                role = 'Human' if isinstance(m, HumanMessage) else 'AI'
            content = str(m.content).replace('\n', '')
            if cut:
                content = (content[:100] + '...') if len(content) > 100 else content
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _generate_session_id():
        """生成带日期的sessionid"""
        date_part = datetime.now().strftime("%Y%m%d")
        random_part = uuid.uuid4().hex[:8]
        return f"{date_part}-{random_part}"

    @staticmethod
    def _check_session_id(session_id: str):
        """检查是否为正确的sessionid"""
        pattern = r'^\d{8}-[0-9a-f]{8}$'
        return bool(re.fullmatch(pattern, session_id))

    @staticmethod
    def _init_checkpointer(db: str):
        if db == "mongodb":
            mongodb_client = MongoClient(str(os.getenv("MONGO_SHORTMEMORY_URL")))
            return MongoDBSaver(mongodb_client, db_name='agentbox')
        # elif db == "redis":
        #     return RedisSaver(redis_url=str(os.getenv("REDIS_SHORTMEMORY_URL")))
        else:
            raise ValueError("暂不支持其他数据库")

    @staticmethod
    def _init_db(db: str = 'mongodb'):
        if db == "mongodb":
            mongodb_client = MongoClient(str(os.getenv("MONGO_SHORTMEMORY_URL")))
            _db = mongodb_client["agentbox"]
            return _db["checkpoints"]
        raise ValueError("暂不支持其他数据库")


def main(args):
    if args.debug:
        print("已启动调试模式")
    agent_box = AgentBox(debug=args.debug)
    agent_box.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False, help="启用调试模式（使用普通 print 输出）")
    main(parser.parse_args())
