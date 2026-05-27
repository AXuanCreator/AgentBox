#!/user/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union
from langchain.tools import tool

import core.impl.sheet_impl as sheet_impl


@tool
def tool_get_csv_excel_path(dir_path: str, depth: int = 1) -> dict:
    """
    获取指定目录下所有的csv和excel，包括子目录，深度默认为1
    :param dir_path: 目录路径
    :param depth: 探索深度
    :return: dict：结果或错误信息
    """
    return sheet_impl.get_csv_excel_path(dir_path, depth)


@tool
def tool_get_columns(dir_path: str) -> dict:
    """
    获取csv或excel的表头
    :param dir_path: csv或excel的文件路径
    :return: dict：结果或错误信息
    """
    return sheet_impl.get_columns(dir_path)


@tool
def tool_get_columns_content(dir_path: str, column: str) -> dict:
    """
    获取某一列的所有内容
    :param dir_path: csv或excel的文件路径
    :param column: 指定的表头字段名
    :return: dict：结果或错误信息
    """
    return sheet_impl.get_columns_content(dir_path=dir_path, column=column)


@tool
def tool_get_row_content(dir_path: str, row: list, sort_mode: str = None) -> dict:
    """
    从CSV或Excel文件中获取指定行的所有列内容，返回JSON格式数据，直接输出或用于下一步处理。
    :param dir_path: 文件路径，支持CSV或Excel格式。
    :param row: 行号列表（1-based，从数据第一行开始计数，不含表头）。
      - 长度1: [a]，获取第a行，适用于用户指定获取单行内容的情况。
      - 长度2: [a, b]，获取第a到b行（包含a和b），如获取1~200行内容则传入[1, 200]
    :param sort_mode: 排序方式，格式为"列名|asc"或"列名|desc"（默认None，不排序）。
      示例: "id|desc"（id列降序）、"like|asc"（like列升序）。
    :return: dict：结果或错误信息
    """
    return sheet_impl.get_row_content(dir_path, row, sort_mode)


@tool
def tool_count_value_in_column(dir_path: str, column: str, value: Union[str, int, float]) -> dict:
    """
    统计表格指定列中某个值出现了多少行（出现频次）。
    适用于：想知道某个特定值在列中出现了多少次。
    例如：「等级」列中 "优秀" 出现了几次 → value="优秀" → 返回 {"count": 12}

    :param dir_path: 表格文件路径（csv或xlsx）
    :param column: 要统计的列名
    :param value: 要统计的具体值（字符串或数字）
    :return: dict：结果或错误信息
    """
    return sheet_impl.count_value_in_column(dir_path=dir_path, column=column, value=value)


@tool
def tool_calculate_add(values: list[Union[int, float]]) -> dict:
    """
    对列表中的元素进行相加
    :param values: 元素列表，元素类型可为int或float
    :return: dict：结果或错误信息
    """
    return sheet_impl.calculate_add(values=values)


@tool
def tool_count_data_rows(dir_path: str) -> dict:
    """
    统计有多少数据行，数据行即不包括表头的有效行数
    :param dir_path: csv或excel的文件路径
    :return: dict：结果或错误信息
    """
    return sheet_impl.count_data_rows(dir_path=dir_path)


@tool
def tool_write_to_table(data: list[list[str]], file_path: str, columns: list = None) -> dict:
    """
    将特定内容全新写入或追加到指定的xlsx中
    :param data: 数据行，形式应为 [[数据1, 数据2, 数据3], [数据4, 数据5, 数据6]]，注意每一行的长度应当与columns长度一致
    :param file_path: 表格路径，支持csv或xlsx，若为全新写入类型的任务，优先使用xlsx
    :param columns:表格的表头，形式应为 [列名1, 列名2, 列名3]，注意应当与data中每一行的长度一致
    :return: dict：结果或错误信息
    """
    return sheet_impl.write_to_table(data=data, file_path=file_path, columns=columns)
