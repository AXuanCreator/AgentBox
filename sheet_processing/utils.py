#!/user/bin/env python3
# -*- coding: utf-8 -*-
import os
from typing import List, Dict, Union

import pandas as pd
from pandas import DataFrame


def get_csv_excel_path(dir_path: str, depth: int = 1) -> Union[List, str]:
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

    return csv_files


def get_columns_content(dir_path: str, column: str) -> Union[List, str]:
    """获取指定列内容"""
    df = read_csv_excel(dir_path)
    if not isinstance(df, DataFrame):
        return str(df)

    df_cols = df.columns.to_list()
    if column not in df_cols:
        return "不存在该表头，检查后重试"

    col_content = df[column].to_list()
    return col_content


def read_csv_excel(dir_path: str) -> Union[DataFrame, str]:
    """读取文件为dataframe"""
    filename = os.path.basename(dir_path)
    _, suffix = os.path.splitext(filename)

    try:
        if suffix == '.csv':
            df = pd.read_csv(dir_path, low_memory=False)
        elif suffix == '.xlsx':
            df = pd.read_excel(dir_path)
        else:
            return "错误：文件类型错误或路径错误，检查后再试"
    except Exception as e:
        return f"错误：{e}"

    return df


def get_columns(dir_path: str) -> Union[List, str]:
    """获取表头内容"""
    df = read_csv_excel(dir_path)
    if not isinstance(df, DataFrame):
        return str(df)
    return df.columns.to_list()


def count_value_in_column(dir_path: str, column: str, value: Union[str, int, float]) -> Union[int, str]:
    """某一列中指定元素出现个数"""
    col_content = get_columns_content(dir_path=dir_path, column=column)
    if not isinstance(col_content, list):
        return "错误：获取列内容失败"
    return col_content.count(value)


def calculate_add(values: List[Union[int, float]]) -> Union[int, float, str]:
    """加法器"""
    try:
        return sum(values)
    except Exception as e:
        return f"错误：{e}"


def get_row_content(dir_path: str, row: List, sort_mode: str = None, writer=None) -> Union[Dict, str]:
    """获取某一行内容"""
    if len(row) not in [1, 2]:
        return "错误：row参数为列表，且长度只能为1或2"

    df = read_csv_excel(dir_path)
    if not isinstance(df, DataFrame):
        return str(df)

    try:
        if sort_mode:  # 可选排序
            df = df_sort(df, sort_mode)
            if not isinstance(df, DataFrame):
                return str(df)  # df是一个str，表示错误消息
        if len(row) == 1:
            return pd.Series(df.iloc[row[0] - 1]).to_dict()
        else:  # 只可能为长度2
            return df.iloc[row[0] - 1:row[1]].to_dict(orient='index')
    except Exception as e:
        return f"错误：{e}"


def df_sort(df: DataFrame, sort_mode: str) -> Union[DataFrame, str]:
    """列排序"""
    sort_col = sort_mode.split('|')[0]
    sort_order = sort_mode.split('|')[1]

    if sort_col not in df.columns:
        return f"错误：不存在{sort_col}列"
    if sort_order not in ['asc', 'desc']:
        return f"错误：排序方式只能为asc或desc"

    try:
        df = df.sort_values(by=sort_col, ascending=True if sort_order == 'asc' else False)
        return df
    except Exception as e:
        return f"错误：{e}"


def count_data_rows(dir_path: str) -> str:
    """表格中有多少数据行（不包括表头）"""
    df = read_csv_excel(dir_path)
    if not isinstance(df, DataFrame):
        return str(df)

    return str(df.shape[0])
