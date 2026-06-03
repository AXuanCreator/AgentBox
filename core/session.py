#!/user/bin/env python3
# -*- coding: utf-8 -*-
import uuid
import re
import os
import questionary
import sqlite3
from datetime import datetime
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from core.config import config


class SessionManager:
    def __init__(self, renderer, db_type: str = "sqlite"):
        self.db_type = db_type
        self.db_collection = self._init_db()
        self.checkpointer = self._init_checkpointer()
        self.display_renderer = renderer

    def _init_checkpointer(self):
        if self.db_type == "mongodb":
            mongodb_client = MongoClient(config.options.mongodb_session_url)
            return MongoDBSaver(mongodb_client, db_name='agentbox')
        elif self.db_type == "sqlite":
            os.makedirs('./data', exist_ok=True)
            sqlite_client = sqlite3.connect("./data/sqlite_checkpoints.db", check_same_thread=False)
            return SqliteSaver(sqlite_client)
        else:
            raise ValueError("暂不支持其他数据库")

    def _init_db(self):
        if self.db_type == "mongodb":
            mongodb_client = MongoClient(config.options.mongodb_session_url)
            _db = mongodb_client["agentbox"]
            return _db["checkpoints"]
        elif self.db_type == "sqlite":
            sqlite_client = sqlite3.connect("./data/sqlite_checkpoints.db", check_same_thread=False)
            return sqlite_client
        raise ValueError("暂不支持其他数据库")

    def get_session_ids(self):
        """从会话记忆数据库中获取所有符合条件的sessionid"""
        if self.db_type == 'mongodb':
            cursor = self.db_collection.find({}, {"thread_id": 1,
                                                  "_id": 0})  # 这里的参数表示只要thread_id字段，而不需要_id字段，即返回数据库中所有thread_id内容
            all_ids = [doc["thread_id"] for doc in cursor if "thread_id" in doc]
        elif self.db_type == 'sqlite':
            cursor = self.db_collection.execute("SELECT DISTINCT thread_id FROM checkpoints")  # 这里已做去重（DISTINCT）
            rows = cursor.fetchall()
            all_ids = [row[0] for row in rows if row[0]]
        else:
            raise ValueError(f"不支持该数据库类型: {self.db_type}")

        pattern = re.compile(r'^\d{8}-')
        return list(set([tid for tid in all_ids if pattern.match(tid)]))

    def check_session_id_available(self, session_id: str):
        """检查该sessionid是否存在于数据库中"""
        session_ids = self.get_session_ids()
        if session_id in session_ids:
            return True
        return False

    def select_session(self):
        session_ids = self.get_session_ids()
        recent5 = "recent_5_sessions"
        if not session_ids:
            self.display_renderer.print_panel("没有找到历史会话", title="📋 会话列表", border_style="yellow")
            return None

        groups: dict[str, list] = {}
        for sid in sorted(session_ids, reverse=True):
            date_part = sid[:8]
            groups.setdefault(date_part, []).append(sid)
        groups.setdefault(recent5, []).extend(sorted(session_ids, reverse=True)[:5])

        self.display_renderer.print('[green]↑↓[/green] 选择，[yellow]Enter[/yellow]确认，[red]Ctrl+C[/red]取消')

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

    @staticmethod
    def generate_session_id():
        """生成带日期的sessionid"""
        date_part = datetime.now().strftime("%Y%m%d")
        random_part = uuid.uuid4().hex[:8]
        return f"{date_part}-{random_part}"

    @staticmethod
    def check_session_id(session_id: str):
        """检查是否为正确的sessionid"""
        pattern = r'^\d{8}-[0-9a-f]{8}$'
        return bool(re.fullmatch(pattern, session_id))





