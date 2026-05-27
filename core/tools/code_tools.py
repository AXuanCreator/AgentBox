#!/user/bin/env python3
# -*- coding: utf-8 -*-

from langchain.tools import tool

from core.impl import code_impl

@tool
def tool_python_executor(code: str) -> dict:
    """
    Python代码执行器，捕获其标准输出，调用者须通过 print 显式输出以获取执行结果
    注意：本工具要求高安全性，不允许破坏用户系统环境，仅当当前工具无法满足用户需求时才可调用
    :param code: 合法的 Python 代码字符串
    :return: dict：结果或错误信息
    """
    return code_impl.python_executor(code)