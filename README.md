# AgentBox

**LangChain驱动的个人助理 Agent，支持表格处理、网页抓取、代码执行与多会话管理**

AgentBox 是一个面向终端的高级 AI 助手框架，主要基于 Langchain 构建。它通过模块化的工具系统与 LLM 协作，能够完成表格数据处理、网页内容抓取、在线搜索、Python 代码执行等多种任务，并提供完整的会话生命周期管理能力。

---


## 功能特性

### 数据表格处理
- 扫描目录下的 CSV / Excel 文件
- 获取表头、列内容、行内容（支持单行与范围）
- 支持按列排序、值计数统计、数据行计数
- 数值列求和计算
- 数据写入与追加

### 网络能力
- 通过 Firecrawl API 将指定 URL 内容转为 Markdown
- 在线搜索查询

### 代码执行
- 安全的 Python 代码执行沙箱，捕获标准输出

### 会话管理
- 自动生成带日期前缀的会话 ID（`yyyymmdd-random`）
- 两级会话选择器（按日期分组 → 选择具体会话）
- 会话切换时自动保留上下文
- 历史消息查看

### 终端体验
- Rich 渲染引擎：Markdown 渲染、彩色面板、Token 用量展示
- 非 TTY 环境自动降级为纯文本模式
- prompt_toolkit 驱动的交互式 REPL，支持多行输入与命令补全
- 自定义快捷键：`Enter` 发送，`Ctrl+J` 换行

### 持久化
- 默认 SQLite 后端，零配置即可运行
- 可选 MongoDB 后端支持

---

## 技术栈

| 层 | 技术 |
|---|---|
| 框架 | LangChain, LangGraph |
| LLM | DeepSeek (ChatDeepSeek), OpenAI 兼容接口 |
| 终端 UI | Rich, prompt_toolkit, questionary |
| 数据处理 | pandas, openpyxl |
| 持久化 | SQLite (默认), MongoDB (可选) |
| 网页抓取 | Firecrawl |
| 架构 | 模块化：config / display / session / history |

---

## 项目结构

```
AgentBox/
├── agent.py                    # 主入口 + REPL 编排 (~175行)
├── keybinding.py               # prompt_toolkit 快捷键绑定
├── .env                        # 环境变量配置
├── core/
│   ├── __init__.py
│   ├── config.py               # AppConfig 配置管理 (pydantic)
│   ├── prompt.py               # SystemPrompts 系统提示词 (dataclass)
│   ├── llm_builder.py          # LLM 工厂 (provider 自动检测)
│   ├── display.py              # 终端渲染 (Rich/Plain 策略)
│   ├── session.py              # 会话生命周期管理
│   ├── history.py              # 历史消息格式化与摘要
│   ├── schemas.py              # ToolResponse / ResponseCode
│   ├── impl/                   # 业务逻辑实现
│   └── tools/                  # @tool 工具定义
└── data/                       # SQLite 持久化数据
```

---

## 快速开始

### 环境要求

- Python 3.10+
- （可选）MongoDB 实例

### 安装依赖

```bash
pip install langchain langchain-openai langchain-deepseek langchain-community langchain-classic langgraph
pip install pandas pymongo python-dotenv
pip install questionary rich openpyxl prompt-toolkit firecrawl-py
```

> **注意**: `langchain-deepseek` v1.0.1 的 `_get_request_payload` 方法需要按 [langchain-ai/langchain#37174](https://github.com/langchain-ai/langchain/issues/37174) 调整，否则会导致 `Missing reasoning_content` 错误。

### 环境变量

编辑 `.env` 文件：

| 变量 | 说明 | 必填 |
|---|---|---|
| `API_KEY` | LLM API 密钥 | 是 |
| `BASE_URL` | LLM API 地址 | 是 |
| `CHAT_MODEL` | 对话模型名称 | 是 |
| `SUMMARY_MODEL` | 摘要模型名称 | 是 |
| `FIRECRAWL_API_KEY` | Firecrawl API 密钥 | 网页抓取时需要 |
| `MONGODB_SESSION_URL` | MongoDB 连接串 | 使用 MongoDB 时需要 |
| `LANGCHAIN_TRACING_V2` | LangSmith 追踪开关 | 否 |
| `LANGCHAIN_API_KEY` | LangSmith API 密钥 | 否 |
| `LANGCHAIN_PROJECT` | LangSmith 项目名称 | 否 |
| `LANGGRAPH_STRICT_MSGPAK` | LangGraph 严格模式 | 建议设为 `true` |

### 运行

```bash
# 正常模式（Rich 终端 UI）
python agent.py

# 调试模式（print 输出，适合 IDE）
python agent.py --debug
```

---

## 可用工具

| 工具 | 功能 |
|---|---|
| `tool_get_csv_excel_path` | 扫描目录下的 CSV / Excel 文件 |
| `tool_get_columns` | 获取表格表头 |
| `tool_get_columns_content` | 获取指定列的全部内容 |
| `tool_get_row_content` | 获取行内容（支持范围与排序） |
| `tool_count_value_in_column` | 统计某列中指定值的出现次数 |
| `tool_count_data_rows` | 统计有效数据行数 |
| `tool_calculate_add` | 对一组数值求和 |
| `tool_write_to_table` | 写入或追加数据到 CSV / Excel |
| `tool_fetch_single_url_to_md` | 抓取网页转为 Markdown |
| `tool_search_online_by_query` | 在线搜索 |
| `tool_python_executor` | 执行 Python 代码并捕获输出 |
| `tool_get_history` | 获取当前会话历史消息 |

---

## 交互命令

| 命令 | 功能 |
|---|---|
| `exit` | 退出并保存当前会话 |
| `/session` | 打开会话选择器 |
| `/session <id>` | 直接切换到指定会话 |
| `/history` | 查看当前会话全部消息 |
| `/clear` 或 `/new` | 清除上下文并创建新会话 |

---

## 响应模型

所有工具统一返回 `ToolResponse` 结构：

```python
class ToolResponse(BaseModel):
    success: bool
    code: ResponseCode    # 标准错误码枚举
    message: str
    data: list | dict | str | None
```

标准错误码涵盖：文件不存在、格式错误、参数错误、数据不匹配、写入错误等场景。

---

## 调试模式

使用 `--debug` 参数启动后，AgentBox 将绕过 Rich 渲染，使用普通 `print` 输出，兼容 PyCharm、VS Code 等 IDE 的内置终端。
