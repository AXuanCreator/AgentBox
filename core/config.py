#!/user/bin/env python3
# -*- coding: utf-8 -*-
import json

import questionary
from pydantic import BaseModel


class ModelsConfig(BaseModel):
    base_url: str
    api_key: str
    main: str
    summary: str
    temperature: float = 0.7
    timeout: int = 600


class OptionsConfig(BaseModel):
    firecrawl_api_key: str
    mongodb_session_url: str | None
    db_type: str = "sqlite"


class AppConfig(BaseModel):
    models: ModelsConfig
    options: OptionsConfig
    model_config = {"extra": "ignore"}

    @classmethod
    def from_json(cls, path: str = "config.json") -> "AppConfig":
        with open(path, encoding='utf-8') as f:
            return cls.model_validate(json.load(f))


class ConfigManager:
    def __init__(self, app_config: AppConfig, renderer):
        self.app_config = app_config.model_dump()
        self.display_renderer = renderer

    def select_config(self):
        first_choices = []

        for k in self.app_config:
            first_choices.append(questionary.Choice(
                title=k,
                value=k
            ))
        self.display_renderer.print('[green]↑↓[/green] 选择，[yellow]Enter[/yellow]确认，[red]Ctrl+C[/red]取消')
        first_selected = questionary.select(
            "一级配置项",
            choices=first_choices,
            use_indicator=True,
            instruction=''
        ).ask()
        if first_selected is None:
            return None, None

        second_items = self.app_config.get(first_selected)
        second_choices = [
            questionary.Choice(
                title=f"{k} -> {v}",
                value=k
            )
            for k, v in second_items.items()
        ]
        second_selected = questionary.select(
            "二级配置项",
            choices=second_choices,
            use_indicator=True,
            instruction=''
        ).ask()
        if second_selected is None:
            return None, None

        return first_selected, second_selected

    def edit_config(self, first_key: str, second_key: str):
        new_value = questionary.text(
            f"{first_key}.{second_key}: ",
            default=str(self.app_config[first_key][second_key]),
        ).ask()
        if new_value is None:
            return None, False

        self.app_config[first_key][second_key] = new_value
        self.app_config = AppConfig.model_validate(self.app_config).model_dump()  # 检查输入的值是否符合特定类型

        with open("config.json", "w", encoding='utf-8') as f:
            json.dump(self.app_config, f, indent=2, ensure_ascii=False)

        return new_value, True

    @staticmethod
    def reload_config():
        new_config = AppConfig.from_json()
        config.models = new_config.models
        config.options = new_config.options

config = AppConfig.from_json()
