#!/user/bin/env python3
# -*- coding: utf-8 -*-
from typing import List, Dict, Union
from langchain.tools import tool, ToolRuntime

import sheet_processing.utils as sheet_utils


@tool
def tool_get_csv_excel_path(dir_path: str, depth: int = 1) -> Union[List, str]:
    """
    获取指定目录下所有的csv和excel，包括子目录，深度默认为1
    :param dir_path: 目录路径
    :param depth: 探索深度
    :return: csv文件路径列表
    """
    return sheet_utils.get_csv_excel_path(dir_path, depth)


@tool
def tool_get_columns(dir_path: str) -> Union[List, str]:
    """
    获取csv或excel的表头
    :param dir_path: csv或excel的文件路径
    :return: List：表头字段列表，str：错误或警告信息
    """
    return sheet_utils.get_columns(dir_path)


@tool
def tool_get_columns_content(dir_path: str, column: str) -> Union[List, str]:
    """
    获取某一列的所有内容
    :param dir_path: csv或excel的文件路径
    :param column: 指定的表头字段名
    :return: List：列内容，str：错误或警告信息
    """
    return sheet_utils.get_columns_content(dir_path=dir_path, column=column)


@tool
def tool_get_row_content(dir_path: str, row: List, runtime: ToolRuntime, sort_mode: str = None) -> Union[Dict, str]:
    """
    从CSV或Excel文件中获取指定行的所有列内容，返回JSON格式数据，直接输出或用于下一步处理。
    :param dir_path: 文件路径，支持CSV或Excel格式。
    :param row: 行号列表（1-based，从数据第一行开始计数，不含表头）。
      - 长度1: [a]，获取第a行，适用于用户指定获取单行内容的情况。
      - 长度2: [a, b]，获取第a到b行（包含a和b），如获取1~200行内容则传入[1, 200]
    :param runtime: LangChain自动注入的运行时上下文（忽略此参数）。
    :param sort_mode: 排序方式，格式为"列名|asc"或"列名|desc"（默认None，不排序）。
      示例: "id|desc"（id列降序）、"like|asc"（like列升序）。
    :return: dict，包含行数据的JSON对象（键为行号，值为该行字典）；或str（错误/警告信息）。
    """
    writer = runtime.stream_writer
    return sheet_utils.get_row_content(dir_path, row, sort_mode, writer)


@tool
def tool_count_value_in_column(dir_path: str, column: str, values: List) -> Union[Dict, str]:
    """
    统计指定列中某个元素（value）出现的次数
    :param dir_path: csv或excel的文件路径
    :param column: 指定的表头字段名
    :param values: 指定的元素名字列表
    :return: Dict：字典{查找值: 出现次数}；str：错误或警告信息
    """
    result = {}
    for value in values:
        result[value] = sheet_utils.count_value_in_column(dir_path=dir_path, column=column, value=value)
    return result


@tool
def tool_calculate_add(values: List[Union[int, float]]) -> Union[int, float, str]:
    """
    对列表中的元素进行相加
    :param values: 元素列表，元素类型可为int或float
    :return: int、float：相加结果，str：错误或警告信息
    """
    return sheet_utils.calculate_add(values=values)


@tool
def tool_count_data_rows(dir_path: str) -> str:
    """
    统计有多少数据行，数据行即不包括表头的有效行数
    :param dir_path: csv或excel的文件路径
    :return: str：行数、错误或警告信息
    """
    return sheet_utils.count_data_rows(dir_path=dir_path)