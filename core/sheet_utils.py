#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
from typing import Union

import pandas as pd
from pandas import DataFrame

from core.schemas import ResponseCode, ToolResponse


def _read_csv_excel(dir_path: str) -> Union[DataFrame, dict]:
    """读取文件为dataframe"""
    filename = os.path.basename(dir_path)
    _, suffix = os.path.splitext(filename)

    try:
        if suffix == '.csv':
            df = pd.read_csv(dir_path, low_memory=False)
        elif suffix == '.xlsx':
            df = pd.read_excel(dir_path)
        else:
            return ToolResponse(success=False, code=ResponseCode.FILE_NOT_FOUND, message=f"文件类型错误或路径错误", data=None).model_dump()
    except Exception as e:
        return ToolResponse(success=False, code=ResponseCode.GENERIC_ERROR, message=f"错误：{e}", data=None).model_dump()

    return df


def _df_sort(df: DataFrame, sort_mode: str) -> Union[DataFrame, dict]:
    """列排序"""
    sort_col = sort_mode.split('|')[0]
    sort_order = sort_mode.split('|')[1]

    if sort_col not in df.columns:
        return ToolResponse(success=False, code=ResponseCode.HEADER_ERROR, message=f"不存在{sort_col}列", data=None).model_dump()
    if sort_order not in ['asc', 'desc']:
        return ToolResponse(success=False, code=ResponseCode.PARAM_ERROR, message=f"排序方式只能为asc或desc", data=None).model_dump()

    try:
        df = df.sort_values(by=sort_col, ascending=True if sort_order == 'asc' else False)
        return df
    except Exception as e:
        return ToolResponse(success=False, code=ResponseCode.GENERIC_ERROR, message=f"错误：{e}", data=None).model_dump()


def get_csv_excel_path(dir_path: str, depth: int = 1) -> dict:
    """获取指定目录下的表格"""
    csv_files = []
    for root, dirs, files in os.walk(dir_path):
        current_depth = root[len(dir_path):].count(os.sep)  # 计算当前深度
        if current_depth > depth:
            dirs.clear()
            continue
        for file in files:
            if file.endswith('.csv') or file.endswith('.xlsx'):
                csv_files.append(os.path.join(root, file))

    return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"计算成功", data=csv_files).model_dump()


def get_columns_content(dir_path: str, column: str) -> dict:
    """获取指定列内容"""
    df = _read_csv_excel(dir_path)
    if isinstance(df, dict):
        return df

    df_cols = df.columns.to_list()
    if column not in df_cols:
        return ToolResponse(success=False, code=ResponseCode.HEADER_ERROR, message=f"不存在该表头", data=None).model_dump()

    col_content = df[column].to_list()
    return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"获取指定列内容成功", data=col_content).model_dump()


def get_columns(dir_path: str) -> dict:
    """获取表头内容"""
    df = _read_csv_excel(dir_path)
    if isinstance(df, dict):
        return df
    return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"获取表头成功", data=df.columns.to_list()).model_dump()


def count_value_in_column(dir_path: str, column: str, value: Union[str, int, float]) -> dict:
    """二级-某一列中指定元素出现个数"""
    col_content = get_columns_content(dir_path=dir_path, column=column)  # 该函数为工具直接依赖函数，返回类型固定为dict
    if not col_content['success']:
        return col_content
    col_count = col_content['data'].count(value)
    return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"获取出现个数成功", data=col_count).model_dump()


def calculate_add(values: list[Union[int, float]]) -> dict:
    """加法器"""
    try:
        return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"计算成功", data=str(sum(values))).model_dump()
    except Exception as e:
        return ToolResponse(success=False, code=ResponseCode.GENERIC_ERROR, message=f"错误：{e}", data=None).model_dump()


def get_row_content(dir_path: str, row: list, sort_mode: str = None) -> dict:
    """获取某一行内容"""
    if len(row) not in [1, 2]:
        return ToolResponse(success=False, code=ResponseCode.PARAM_ERROR, message=f"row参数应为列表，单行查询长度为1，多行范围查询长度为2", data=None).model_dump()

    df = _read_csv_excel(dir_path)
    if isinstance(df, dict):
        return df

    try:
        if sort_mode:  # 可选排序
            df = _df_sort(df, sort_mode)
            if isinstance(df, dict):
                return df
        if len(row) == 1:
            return pd.Series(df.iloc[row[0] - 1]).to_dict()
        else:  # 只可能为长度2
            return df.iloc[row[0] - 1:row[1]].to_dict(orient='index')
    except Exception as e:
        return ToolResponse(success=False, code=ResponseCode.GENERIC_ERROR, message=f"错误：{e}", data=None).model_dump()


def count_data_rows(dir_path: str) -> dict:
    """表格中有多少数据行（不包括表头）"""
    df = _read_csv_excel(dir_path)
    if isinstance(df, dict):
        return df

    return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"获取行数成功", data=str(df.shape[0])).model_dump()


def write_to_table(data: list[list[str]], file_path: str, columns: list = None) -> dict:
    # TODO: 这里应该需要权限管理
    if os.path.exists(file_path):
        if columns is not None and columns != get_columns(file_path):
            return ToolResponse(success=False, code=ResponseCode.HEADER_ERROR, message=f"该文件已存在，而传入的表头内容与该文件表头不一样", data=None).model_dump()
        if not columns:
            origin_columns = get_columns(file_path)
            if not origin_columns['success']:
                return origin_columns
            columns = origin_columns['data']
        df = _read_csv_excel(file_path)
        if isinstance(df, dict):
            return df
    else:
        if not columns:
            return ToolResponse(success=False, code=ResponseCode.PARAM_ERROR, message=f"写入excel时需要表头列表", data=None).model_dump()
        df = pd.DataFrame(columns=columns)

    # 检查data与columns的长度
    if any(len(row) != len(columns) for row in data):
        return ToolResponse(success=False, code=ResponseCode.PARAM_ERROR, message=f"数据行列数与表头列数不一致", data=None).model_dump()

    new_rows = pd.DataFrame(data=data, columns=columns)
    df = pd.concat([df, new_rows], ignore_index=True)

    ext = os.path.splitext(file_path)[-1]
    try:
        if ext == '.csv':
            df.to_csv(file_path, index=False)
        elif ext == '.xlsx':
            df.to_excel(file_path, index=False)
        else:
            return ToolResponse(success=False, code=ResponseCode.PARAM_ERROR, message=f"错误：file_path只支持csv或xlsx后缀的文件路径", data=None).model_dump()
    except Exception as e:
        return ToolResponse(success=False, code=ResponseCode.GENERIC_ERROR, message=f"错误：{e}", data=None).model_dump()

    return ToolResponse(success=True, code=ResponseCode.SUCCESS, message=f"写入成功", data=None).model_dump()
