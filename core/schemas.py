#!/user/bin/env python3
# -*- coding: utf-8 -*-

from enum import Enum
from typing import Union
from pydantic import BaseModel

_constructing = False


class ResponseCode(str, Enum):
    SUCCESS = "SUCCESS"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"  # 文件不存在
    INVALID_FORMAT = "INVALID_FORMAT"  # 格式错误
    HEADER_ERROR = "HEADER_ERROR"  # 表头错误
    PARAM_ERROR = "PARAM_ERROR"  # 参数不可用
    DATA_ERROR = "DATA_ERROR"  # 数据不匹配
    URL_ERROR = "URL_ERROR"  # URL格式错误
    HISTORY_ERROR = "HISTORY_ERROR"  # 历史消息型错误
    WRITE_ERROR = "WRITE_ERROR"  # 写入错误
    GENERIC_ERROR = "GENERIC_ERROR"  # 通常错误


class ToolResponse(BaseModel):
    success: bool
    code: ResponseCode
    message: str
    data: Union[list, dict, str] | None


if __name__ == '__main__':
    print(ToolResponse(success=True, code=ResponseCode.SUCCESS, message='Success', data='data'))
