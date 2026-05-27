#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
import subprocess
import textwrap

from core.schemas import ResponseCode, ToolResponse

dotenv_path = ".env"
load_dotenv(dotenv_path=dotenv_path, override=True)


def python_executor(code: str) -> dict:
    try:
        env = os.environ.copy()
        result = subprocess.check_output(
            ["python", "-c", textwrap.dedent(code)],
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"执行代码成功", data=result).model_dump()
    except Exception as e:
        return ToolResponse(success=False, code=ResponseCode.GENERIC_ERROR, message=f"错误：{e}", data=None).model_dump()


if __name__ == '__main__':
    x = python_executor(
        """
        result = sum(range(1, 101))  # 计算1到100的和
        print(result)
        """
    )
    print()