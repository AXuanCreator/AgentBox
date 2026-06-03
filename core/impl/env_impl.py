#!/user/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import textwrap
import platform

from pathlib import Path

from core.schemas import ResponseCode, ToolResponse


def get_system_plat() -> dict:
    """获取系统信息（windows/linux）"""
    system_plat = platform.system()
    return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"获取系统信息成功", data=system_plat).model_dump()


def python_executor(code: str) -> dict:
    try:
        result = subprocess.run(
            ["python", "-c", textwrap.dedent(code)],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"执行代码成功", data=result.stdout or result.stderr).model_dump()
    except Exception as e:
        return ToolResponse(success=False, code=ResponseCode.GENERIC_ERROR, message=f"错误：{e}", data=None).model_dump()


def shell_executor(cmd: str) -> dict:
    try:
        result = subprocess.run(
            cmd.strip(),
            shell=True,
            capture_output=True,
            timeout=120,
            errors="replace",
        )
        return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"执行指令成功", data=result.stdout or result.stderr).model_dump()
    except Exception as e:
        return ToolResponse(success=False, code=ResponseCode.GENERIC_ERROR, message=f"错误：{e}", data=None).model_dump()


def file_read_text(path: str) -> dict:
    try:
        content = Path(path.strip()).read_text(encoding='utf-8')
        return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"读取文件成功", data=content).model_dump()
    except Exception as e:
        return ToolResponse(success=False, code=ResponseCode.GENERIC_ERROR, message=f"错误：{e}", data=None).model_dump()


def file_write_text(path: str, content: str, write_mode: str) -> dict:
    try:
        if write_mode not in ['w', 'a']:
            return ToolResponse(success=False, code=ResponseCode.PARAM_ERROR, message=f"错误：write_mode不支持除w或a之外的字符串", data=None).model_dump()
        with Path(path.strip()).open(write_mode, encoding='utf-8') as f:
            response = f.write(content)
        return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"写入文件成功", data=f"写入字符串数：{response}").model_dump()
    except Exception as e:
        return ToolResponse(success=False, code=ResponseCode.GENERIC_ERROR, message=f"错误：{e}", data=None).model_dump()


def file_exists(path: str) -> dict:
    try:
        result = Path(path.strip()).is_file()
        return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"获取文件存在状态成功", data=str(result)).model_dump()
    except Exception as e:
        return ToolResponse(success=False, code=ResponseCode.GENERIC_ERROR, message=f"错误：{e}", data=None).model_dump()


def dir_exists(path: str) -> dict:
    try:
        result = Path(path.strip()).is_dir()
        return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"获取文件夹存在状态成功", data=str(result)).model_dump()
    except Exception as e:
        return ToolResponse(success=False, code=ResponseCode.GENERIC_ERROR, message=f"错误：{e}", data=None).model_dump()


def get_working_dir() -> dict:
    try:
        result = Path.cwd()
        return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"获取工作目录成功", data=str(result)).model_dump()
    except Exception as e:
        return ToolResponse(success=False, code=ResponseCode.GENERIC_ERROR, message=f"错误：{e}", data=None).model_dump()


if __name__ == '__main__':
    x = shell_executor(
        """
        dir -a
        """
    )
    print()
