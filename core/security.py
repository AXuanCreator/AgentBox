#!/user/bin/env python3
# -*- coding: utf-8 -*-
import threading

import questionary
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm_builder import init_llm
from core.config import config
from core.prompt import system_prompts


class SecurityReviewer:
    def __init__(self):
        self.display_renderer = None
        self.status_obj = None
        security_reviewer_fast = init_llm(config.security.model_fast, config.security.base_url, config.security.api_key)
        security_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompts.security_prompt),
            ("human", "{messages_str}")
        ])
        self.security_chain = security_prompt | security_reviewer_fast | StrOutputParser()
        self._selector_lock = threading.Lock()

    def set_display_renderer(self, display_renderer):
        self.display_renderer = display_renderer

    def set_status_obj(self, status_obj):
        self.status_obj = status_obj

    def security_confirm_selector(self, content):
        """返回权限选择器"""
        with self._selector_lock:
            if self.status_obj:
                self.status_obj.stop()
            try:
                choices = [
                    questionary.Choice(title="Yes", value="yes"),
                    questionary.Choice(title="No", value="no"),
                    questionary.Choice(title="Enter More Info", value="info")
                ]
                self.display_renderer.print_panel(
                    content=content,
                    title="执行内容",
                    border_style="orange_red1"
                )
                self.display_renderer.print('[green]↑↓[/green] 选择，[yellow]Enter[/yellow]确认，[red]Ctrl+C[/red]取消')
                selected = questionary.select(
                    "是否执行",
                    choices=choices,
                    use_indicator=True,
                    instruction=''
                ).ask()
                if selected == "info":
                    new_value = questionary.text(
                        f"提供更多信息："
                    ).ask()
                    return new_value or "no"  # Ctrl+C时返回no
                return selected or "no"
            finally:
                if self.status_obj:
                    self.status_obj.start()

    def security_check(self, content):
        """使用llm对指令进行安全审查"""
        response = self.security_chain.invoke({'messages_str': content})
        return response


security_reviewer = SecurityReviewer()
