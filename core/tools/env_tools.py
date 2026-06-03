#!/user/bin/env python3
# -*- coding: utf-8 -*-

from langchain.tools import tool

from core.impl import env_impl


@tool
def tool_get_system_plat() -> dict:
    """
    获取当前系统平台（Windows、Linux、Darwin等）信息
    :return: dict：结果或错误信息
    """
    return env_impl.get_system_plat()


@tool
def tool_python_executor(code: str) -> dict:
    """
    Python代码执行器，捕获其标准输出，调用者须通过 print 显式输出以获取执行结果
    注意：本工具要求高安全性，不允许代码破坏用户系统环境，仅当当前已有工具无法满足用户需要时才可调用
    :param code: 合法的 Python 代码字符串
    :return: dict：结果或错误信息
    """
    return env_impl.python_executor(code)


@tool
def tool_shell_executor(cmd: str) -> dict:
    """
    系统指令执行器，捕获其标准输出
    注意：本工具要求高安全性，不允许指令破坏用户系统环境，仅当当前已有工具无法满足用户需要时才可调用
    :param cmd: 合法的 系统指令 字符串
    :return: dict：结果或错误信息
    """
    return env_impl.shell_executor(cmd)


@tool
def tool_file_read_text(path: str) -> dict:
    """
    文件读取工具，可读取所有文本格式的文件信息，不支持二进制文件
    :param path: 文件路径
    :return: dict：结果或错误信息
    """
    return env_impl.file_read_text(path)


@tool
def tool_file_write_text(path: str, content: str, write_mode: str) -> dict:
    """
    文件写入工具，要求提供写入类型（覆盖、追加）
    注意：本工具要求高安全性，不允许指令破坏用户系统环境，不允许覆写重要文件
    :param path: 文件路径
    :param content: 待写入内容
    :param write_mode: 写入类型，可为'w'或'a'，其中w表示覆盖式写入，a表示追加式写入
    :return: dict：结果或错误信息
    """
    return env_impl.file_write_text(path, content, write_mode)


@tool
def tool_file_exists(path: str) -> dict:
    """
    查看指定路径文件是否存在
    :param path: 文件路径
    :return: dict：结果或错误信息
    """
    return env_impl.file_exists(path)


@tool
def tool_dir_exists(path: str) -> dict:
    """
    查看指定路径文件夹是否存在
    :param path: 文件夹路径
    :return: dict：结果或错误信息
    """
    return env_impl.dir_exists(path)


@tool
def tool_get_working_dir() -> dict:
    """
    查看当前工作目录
    :return: dict：结果或错误信息
    """
    return env_impl.get_working_dir()
