<div align="center">
    <h1>AIDGENT</h1>
    <p><strong>一个具备安全感知代码执行与多会话管理能力的模块化个人助理智能体</strong></p>
</div>

<div align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://www.langchain.com/">
    <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain">
  </a>
  <a href="https://langchain-ai.github.io/langgraph/">
    <img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangGraph">
  </a>
</div>



<p align="center">
  <a href="#摘要">摘要</a> •
  <a href="#特性">特性</a> •
  <a href="#系统架构">系统架构</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#工具系统">工具系统</a> •
  <a href="#安全机制">安全机制</a> •
  <a href="#交互命令">交互命令</a> 
</p>
---

## 摘要

Aidgent 是一个开源的、基于大语言模型（LLM）的自主智能体框架，运行于终端环境之中。它将 LangChain/LangGraph 的智能体运行时与一套精心设计的工具系统相结合，使 LLM 能够直接操作本地文件、电子表格、代码执行环境以及网络资源。Aidgent 提供了双模式渲染引擎（Rich TTY 与纯文本）、热重载配置系统，以及支持 SQLite 和 MongoDB 两种后端的持久化多会话管理层。

---

## 特性

### 自主性

Aidgent 能够通过对可用工具的推理，自主分解并执行复杂任务。智能体从网络抓取到数据分析再到文件操作，自动选择合适的工具链，无需人工干预即可完成常规操作。

### 安全感知执行
每一次 Python 和 Shell 执行均通过专用安全审查 LLM 进行分类。`unsafe` 操作被直接阻止，`confirm` 操作需经用户明确批准，`non-direct` 操作则被标记以供进一步审查。

### 多会话持久化
所有智能体会话通过 LangGraph 的检查点机制自动持久化。用户可在历史会话间切换、回顾过往交互，并在任意时刻恢复工作。

### 双模式终端渲染
基于策略模式的渲染层，根据运行环境自动选择 Rich TTY 输出或纯文本调试输出，确保在原生终端和 IDE 集成控制台中均可正常使用。

### 运行时配置
包括 LLM 端点、模型选择和执行选项在内的配置参数，可在终端内交互式查看和编辑。模型相关变更通过热重载即时生效，无需重启。

---

## 系统架构

Aidgent 采用分层架构，各层职责明确：

| 层级 | 组件 | 职责 |
|---|---|---|
| **编排层** | `agent.py` | REPL 循环、指令分发、流式处理 |
| **智能体运行时** | LangGraph `create_agent` | 工具使用循环、基于检查点的状态持久化 |
| **工具系统** | `core/tools/` | 自动发现的 `@tool` 函数；当前共 14 个工具，分属 4 个类别 |
| **实现层** | `core/impl/` | 纯业务逻辑：表格操作、网页抓取、代码执行、文件 I/O |
| **安全层** | `core/security.py` | 基于 LLM 的安全审查与人在回路确认机制 |
| **会话层** | `core/session.py` | SQLite/MongoDB 双后端会话管理与检查点维护 |
| **显示层** | `core/display.py` | 双策略渲染：`RichRenderer`（TTY）/ `PlainRenderer`（调试） |
| **配置层** | `core/config.py` | Pydantic 验证的配置模型，支持交互式编辑与热重载 |

智能体在标准 LangGraph `create_agent` 循环中运行：LLM 接收系统提示词与会话历史，输出文本响应或工具调用指令，工具执行结果反馈到循环中，如此反复直至任务完成。

---

## 快速开始

### 环境要求

- Python 3.10+
- 兼容的 LLM API 端点（OpenAI 兼容或 DeepSeek）
- （可选）Firecrawl API 密钥，用于网页抓取与搜索

```bash
# 核心智能体框架
pip install langchain langchain-openai langchain-deepseek langchain-community langgraph
# 数据处理
pip install pandas openpyxl
# 终端 UI
pip install rich prompt-toolkit questionary
# 持久化与网络
pip install pymongo firecrawl-py python-dotenv
```

### 配置

复制 `config.example.json` 为 `config.json` 并填入您的凭证信息：

```json
{
  "models": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-xxxx",
    "main": "gpt-4o",
    "summary": "gpt-4o-mini",
    "reasoning_effort": "medium",
    "temperature": 0.7,
    "timeout": 600
  },
  "security": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-xxxx",
    "model_fast": "gpt-4o-mini"
  },
  "options": {
    "firecrawl_api_key": "fc-xxxx",
    "db_type": "sqlite",
    "mongodb_session_url": null
  }
}
```

| 参数 | 说明 |
|---|---|
| `models.base_url` | LLM API 端点地址 |
| `models.api_key` | API 认证密钥 |
| `models.main` | 主智能体模型 |
| `models.summary` | 会话摘要轻量模型 |
| `models.reasoning_effort` | 推理深度：`minimal` / `low` / `medium` / `high` / `xhigh` |
| `security.model_fast` | 安全审查轻量模型（延迟敏感） |
| `options.db_type` | 会话存储后端：`sqlite` 或 `mongodb` |
| `options.firecrawl_api_key` | Firecrawl API 密钥，用于网页抓取与搜索 |

### 运行

```bash
# 正常模式（Rich 终端 UI）
python agent.py

# 调试模式（纯文本输出，适用于 IDE 内置终端）
python agent.py --debug
```

---

## 工具系统

Aidgent 提供 14 个内置工具，分为四个类别。所有工具遵循统一的响应模式（`ToolResponse`），使用标准化状态码。

### 表格工具

| 工具 | 说明 |
|---|---|
| `tool_get_csv_excel_path` | 扫描目录下所有 CSV/Excel 文件 |
| `tool_get_columns` | 获取表头信息 |
| `tool_get_columns_content` | 提取指定列的全部内容 |
| `tool_get_row_content` | 获取行内容，支持范围查询与排序 |
| `tool_count_value_in_column` | 统计某列中指定值的出现次数 |
| `tool_count_data_rows` | 统计数据行数（不含表头） |
| `tool_calculate_add` | 数值列表求和 |
| `tool_write_to_table` | 写入或追加数据到 CSV/Excel |

### 网络工具

| 工具 | 说明 |
|---|---|
| `tool_fetch_single_url_to_md` | 抓取单个网页并转换为 Markdown |
| `tool_search_online_by_query` | 在线搜索查询并返回结构化结果 |

### 系统工具

| 工具 | 说明 |
|---|---|
| `tool_get_system_plat` | 获取宿主机操作系统平台 |
| `tool_python_executor` | 执行 Python 代码（含安全审查） |
| `tool_shell_executor` | 执行 Shell 指令（含安全审查，120s 超时） |
| `tool_file_read_text` | 读取文本文件内容 |
| `tool_file_exists` | 检查文件是否存在 |
| `tool_dir_exists` | 检查目录是否存在 |
| `tool_get_working_dir` | 获取当前工作目录 |

### 运行时工具

| 工具 | 说明 |
|---|---|
| `tool_get_history` | 获取当前会话的完整消息历史 |

---

## 安全机制

Aidgent 采用一套新颖的基于 LLM 的代码审查流水线，以降低自动化代码执行中固有的安全风险：

1. **分类**：执行前，轻量级安全审查 LLM 对代码/指令进行评估，输出四种标签之一：
   - `safe` — 无副作用，只读操作，纯计算
   - `unsafe` — 明显的恶意或破坏性行为；执行被拒绝
   - `confirm` — 潜在风险（文件写入、网络调用等）；需人工确认
   - `non-direct` — 通过外部脚本间接执行；智能体被指示获取并审查实际代码

2. **人在回路**：对于归类为 `confirm` 的操作，系统向用户呈现交互式选择器（"是"、"否"或"提供更多信息"），待确认后方可执行。

3. **可审计性**：每次执行均在响应元数据中标注其安全分类，便于对所有代码执行事件进行事后审查。

---

## 交互命令

| 命令 | 说明 |
|---|---|
| `?` | 切换指令帮助面板 |
| `exit` | 退出并持久化当前会话 |
| `/session` | 打开会话选择器（最近 5 个 + 按日期分组） |
| `/session <id>` | 直接切换到指定会话 |
| `/history` | 查看当前会话历史消息 |
| `/clear` / `/new` | 创建新会话（旧会话保留） |
| `/config` | 交互式查看和编辑配置（模型变更热重载） |

---

