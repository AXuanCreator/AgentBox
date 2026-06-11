import sys
import re
import warnings
import argparse
from collections.abc import Callable

from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style

from core.session import SessionManager
from keybinding import bindings_questionary, get_help_mode
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from core.tools import ALL_TOOLS
from core.config import config, ConfigManager
from core.prompt import system_prompts
from core.llm_builder import init_llm
from core.display import init_display_renderer, process_stream_chunk
from core.history import format_messages_to_str

warnings.filterwarnings("ignore", message="Workbook contains no default style")


class AgentBox:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.tools = ALL_TOOLS
        self.command_handlers: dict[str, Callable[[str], bool]] = {
            "/exit": self._handle_exit,
            "exit": self._handle_exit,
            "/session": self._handle_session,
            "/history": self._handle_history,
            "/clear": self._handle_clear,
            "/new": self._handle_clear,
            "/config": self._handle_config,
        }
        self.command_descriptions = {
            "/exit, exit": "退出CLI",
            "/clear, /new": "新建会话",
            "/session, /session [session_id]": "选择特定会话",
            "/history": "查看会话历史",
            "/config": "配置信息查看与编辑"
        }
        self.agent_command = list(self.command_handlers.keys())  # 快捷指令（“/”）
        self.command_completer = WordCompleter(self.agent_command, ignore_case=True, WORD=True)
        self.command_style = Style([
            ('completion-menu', 'fg:black bg:#cccccc'),
            ('completion-menu.completion.current', 'fg:white bg:blue'),  # 选中项颜色
            ('bottom-toolbar', 'fg:#aaaaaa noreverse'),
        ])
        self.total_token = 0
        self.display_renderer = init_display_renderer(debug=self.debug)
        self.session_manager = SessionManager(renderer=self.display_renderer, db_type=config.options.db_type)
        self.config_manager = ConfigManager(app_config=config, renderer=self.display_renderer)
        self.agent_config = RunnableConfig(configurable={
            "thread_id": self.session_manager.generate_session_id()})  # 当前仅用于区分不同会话（session）
        self.agent_chat = self._build_agent(init_llm(config.models.main, config.models.base_url, config.models.api_key, config.models.temperature, config.models.timeout, config.models.reasoning_effort), self.tools, system_prompts.session_launch_prompt, self.session_manager.checkpointer)
        self.llm_summary = init_llm(config.models.summary, config.models.base_url, config.models.api_key, config.models.temperature, config.models.timeout)  # 无提示词，无记忆的总结普通llm

    def _dispatch_command(self, user_input: str) -> bool:
        """匹配/命令"""
        for prefix, handler in self.command_handlers.items():
            if user_input == prefix or user_input.startswith(prefix + " "):
                return handler(user_input)
        return False

    def _get_toolbar(self):
        if get_help_mode():
            lines = []
            for cmd, desc in self.command_descriptions.items():
                lines.append(f"  {cmd:<10s} --{desc}")
            return "\n".join(lines)
        return "(ENTER 发送，CTRL+J 换行，CTRL+C 清空输入内容，? 显示指令面板)"

    def _handle_exit(self, _user_input: str) -> bool:
        self.display_renderer.print_panel(
            f"本次会话已保存，可通过[bright_blue]/session[/bright_blue]或[bright_blue]/session {self.agent_config['configurable']['thread_id']}[/bright_blue]来恢复会话",
            title=f"会话保存 {self.agent_config['configurable']['thread_id']}",
            border_style="dark_red",
        )
        sys.exit()

    def _handle_session(self, user_input: str) -> bool:
        parts = user_input.split(maxsplit=1)
        if len(parts) == 1:
            if self.debug:
                self.display_renderer.print(f"调试模式不支持使用session选择器，请直接输入特定session_id:\n{self.session_manager.get_session_ids()}")
                return True
            selected = self.session_manager.select_session()
        else:
            selected = parts[1]

        if not selected:
            return True
        if not self.session_manager.check_session_id(str(selected)):
            self.display_renderer.print("[bold red]输入的session_id格式错误[/bold red]，正确格式为 [green]yyyymmdd-12345678[/green]")
            return True
        if not self.session_manager.check_session_id_available(str(selected)):
            self.display_renderer.print("[bold red]该session_id不存在，请检查后重试[/bold red]")
            return True

        self.agent_config["configurable"]["thread_id"] = selected
        self.agent_chat = self._build_agent(init_llm(config.models.main, config.models.base_url, config.models.api_key, config.models.temperature, config.models.timeout, config.models.reasoning_effort), self.tools, system_prompts.session_launch_prompt, self.session_manager.checkpointer)
        history_message = self.agent_chat.get_state(self.agent_config).values["messages"]
        # history_summary = summarize_history(history_message, self.llm_summary)
        history_summary = "该功能正在维护..."  # TODO: 历史总结时长太长
        self.display_renderer.print_panel(
            f"已切换到会话: [green]{selected}[/green]\n过往消息总结: {history_summary}\n\n[dim]可通过[bright_blue]/history[/bright_blue]或直接咨询来查询历史消息[/dim]",
            title="✅ 会话切换",
            border_style="green",
        )
        return True

    def _handle_history(self, _user_input: str) -> bool:
        history_message = self.agent_chat.get_state(self.agent_config).values.get("messages", "")
        if history_message:
            history_message_format = format_messages_to_str(messages=history_message, cut=True, style=True)
            self.display_renderer.print_panel(
                f"{history_message_format}",
                title="🕛 历史消息",
                border_style="violet",
            )
        else:
            self.display_renderer.print("[bold red]当前会话不存在任何历史消息[/bold red]")
        return True

    def _handle_clear(self, _user_input: str) -> bool:
        new_session = self.session_manager.generate_session_id()
        old_session = self.agent_config["configurable"]["thread_id"]
        self.agent_config["configurable"]["thread_id"] = new_session
        self.agent_chat = self._build_agent(init_llm(config.models.main, config.models.base_url, config.models.api_key, config.models.temperature, config.models.timeout, config.models.reasoning_effort), self.tools, system_prompts.session_launch_prompt, self.session_manager.checkpointer)
        self.display_renderer.print_panel(
            f"已清除上下文并创建新会话: [green]{new_session}[/green]，就会话可通过[grey50]/session {old_session}[/grey50]恢复",
            title="✅ 会话新建",
            border_style="light_slate_grey",
        )
        return True

    def _handle_config(self, _user_input: str) -> bool:
        first_key, second_key = self.config_manager.select_config()
        if first_key is None or second_key is None:
            return False
        new_value, is_ok = self.config_manager.edit_config(first_key, second_key)
        if is_ok:
            self.config_manager.reload_config()
            if second_key in ['db_type', 'mongodb_session_url']:
                self.display_renderer.print(f'配置{first_key}.{second_key}已更新，需要重新启动以加载新配置 | 新配置: {new_value}')
                return True
            if first_key in ['models']:
                self.agent_chat = self._build_agent(init_llm(config.models.main, config.models.base_url, config.models.api_key, config.models.temperature, config.models.timeout, config.models.reasoning_effort), self.tools, system_prompts.session_launch_prompt, self.session_manager.checkpointer)
                self.llm_summary = init_llm(config.models.summary, config.models.base_url, config.models.api_key, config.models.temperature, config.models.timeout)  # 无提示词，无记忆的总结普通llm
            self.display_renderer.print(f'配置{first_key}.{second_key}已更新，热重载成功 | 新配置: {new_value}')
            return True
        return False

    def run(self):
        while True:
            if self.debug:
                user_input = input("> ")
            else:
                user_input = prompt(
                    ">",
                    multiline=True,
                    key_bindings=bindings_questionary,
                    bottom_toolbar=self._get_toolbar,
                    completer=self.command_completer,
                    style=self.command_style
                )
            user_input = re.sub(r'\n+', '\n', user_input).strip('\n')  # 去除末尾所有换行符

            if self._dispatch_command(user_input):
                continue

            with self.display_renderer.status("Thinking...") as status_obj:
                for chunk in self.agent_chat.stream(
                    input={"messages": [{"role": "user", "content": user_input}]},
                    config=self.agent_config,
                    stream_mode="values",
                ):
                    self.total_token = process_stream_chunk(chunk, self.display_renderer, self.total_token, status_obj)

    @staticmethod
    def _build_agent(llm, tools=None, system_prompt=None, checkpointer=None):
        return create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            # middleware=[self.history_summarize],
            checkpointer=checkpointer,
        )


def main(args):
    if args.debug:
        print("已启动调试模式")
    agent_box = AgentBox(debug=args.debug)
    agent_box.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False, help="启用调试模式（使用普通 print 输出）")
    main(parser.parse_args())
