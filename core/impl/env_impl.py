#!/user/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import textwrap
import platform
from pathlib import Path

from core.schemas import ResponseCode, ToolResponse
from core.security import security_reviewer


def _security_code_process(security_code: str, content: str):
    """对返回的security_code进行处理"""
    if security_code == "unsafe":
        return ToolResponse(success=False, code=ResponseCode.SECURITY_ERROR, message=f"该指令会危及系统安全，不允许执行", data=None).model_dump()
    elif security_code == "non-direct":
        return ToolResponse(success=False, code=ResponseCode.WARNING, message="当前执行内容为间接执行指令，无法直接审查执行内容安全性，请先调用文件读取工具获取该文件的实际内容，然后将读取到的实际内容直接使用执行器执行", data=f"指令/代码内容:{content}").model_dump()
    return ToolResponse(success=False, code=ResponseCode.SECURITY_ERROR, message=f"安全审查发生错误：审查机制返回非法内容", data=security_code).model_dump()


def get_system_plat() -> dict:
    """获取系统信息（windows/linux）"""
    system_plat = platform.system()
    return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"获取系统信息成功", data=system_plat).model_dump()


def python_executor(code: str) -> dict:
    try:
        security_code = security_reviewer.security_check(code).strip().lower()
        if security_code == "confirm":
            confirm_response = security_reviewer.security_confirm_selector(code).strip().lower()
            if confirm_response == "no":
                return ToolResponse(success=False, code=ResponseCode.SECURITY_ERROR, message=f"该指令被用户主动拒绝，终止执行", data=None).model_dump()
            elif confirm_response != "no" and confirm_response != "yes":
                return ToolResponse(success=False, code=ResponseCode.INFO, message=f"用户对该执行内容提供了新的信息，根据新的信息调用新方案", data=f"原代码：{code}\n用户提供信息：{confirm_response}").model_dump()
        elif security_code != "safe":
            return _security_code_process(security_code, code)

        result = subprocess.run(
            ["python", "-c", textwrap.dedent(code)],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"执行代码成功", data=(
                                                                                                           result.stdout or result.stderr) + f"security_code: {security_code}").model_dump()
    except Exception as e:
        return ToolResponse(success=False, code=ResponseCode.GENERIC_ERROR, message=f"错误：{e}", data=None).model_dump()


def shell_executor(cmd: str) -> dict:
    try:
        security_code = security_reviewer.security_check(cmd).strip().lower()
        if security_code == "confirm":
            confirm_response = security_reviewer.security_confirm_selector(cmd).strip().lower()
            if confirm_response == "no":
                return ToolResponse(success=False, code=ResponseCode.SECURITY_ERROR, message=f"该指令被用户主动拒绝，终止执行", data=None).model_dump()
            elif confirm_response != "no" and confirm_response != "yes":
                return ToolResponse(success=False, code=ResponseCode.INFO, message=f"用户对该执行内容提供了新的信息", data=f"原代码：{cmd}\n用户提供信息：{confirm_response}").model_dump()
        elif security_code != "safe":
            return _security_code_process(security_code, cmd)

        result = subprocess.run(
            cmd.strip(),
            shell=True,
            capture_output=True,
            timeout=120,
            errors="replace",
        )
        return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"执行指令成功", data=(result.stdout or result.stderr) + f"security_code: {security_code}").model_dump()
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
