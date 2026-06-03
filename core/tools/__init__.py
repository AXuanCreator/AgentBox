#!/user/bin/env python3
# -*- coding: utf-8 -*-

import inspect
from . import sheet_tools, fetch_tools, env_tools, runtime_tools

ALL_TOOLS = []
for mod in [sheet_tools, fetch_tools, env_tools, runtime_tools]:
    for obj in vars(mod).values():
        if hasattr(obj, "tool_call_schema"):
            ALL_TOOLS.append(obj)