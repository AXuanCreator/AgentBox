# 🤖 AgentBox

**基于 LangChain 的模块化个人助理 Agent——在终端中完成表格处理、网页抓取、代码执行、文件操作与多会话管理。**

<p align="left">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/framework-LangChain-green?logo=langchain" alt="LangChain">
</p>

---

## 是什么

AgentBox 是一个运行在终端里的 AI 助手框架。它将 LangChain Agent 与一套精心设计的工具系统结合起来，让 LLM 可以直接操作你本地的文件、表格、代码和网络资源。AgentBox 提供完善的多会话管理，所有对话自动持久化，随时可切回任意历史会话继续工作。

## 核心能力

<table align="center">
  <tr align="center">
    <td width="33%"><b>📊 数据表格</b><br><sub>CSV / Excel 读写、统计与排序</sub></td>
    <td width="33%"><b>🌐 网络能力</b><br><sub>网页抓取转 Markdown、在线搜索</sub></td>
    <td width="33%"><b>⚡ 代码执行</b><br><sub>Python 与 Shell 指令执行</sub></td>
  </tr>
  <tr align="center">
    <td><b>📁 文件操作</b><br><sub>文件读写、存在性检查、目录查询</sub></td>
    <td><b>💬 会话管理</b><br><sub>多会话持久化、历史切换、摘要</sub></td>
    <td><b>⚙️ 运行时配置</b><br><sub>终端内实时编辑、参数热重载</sub></td>
  </tr>
</table>

## 快速开始

### 环境要求

- Python 3.10+
- （可选）MongoDB 实例

### 安装

```bash
# 核心框架
pip install langchain langchain-openai langchain-deepseek langchain-openrouter langchain-community langchain-classic langgraph
# 数据处理
pip install pandas openpyxl
# 终端 UI
pip install rich prompt-toolkit questionary
# 持久化与网络
pip install pymongo firecrawl-py python-dotenv
```

> **注意**：`langchain-deepseek` v1.0.1 的 `_get_request_payload` 方法需按 [langchain-ai/langchain#37174](https://github.com/langchain-ai/langchain/issues/37174) 调整，否则会导致 `Missing reasoning_content` 错误。

### 配置

复制 `config.example.json` 为 `config.json` 并填入你的信息：

```json
{
  "models": {
    "base_url": "https://api.openai.com/v1",
    "api_key": "sk-xxxx",
    "main": "gpt-4o",
    "summary": "gpt-4o-mini",
    "temperature": 0.7,
    "timeout": 600
  },
  "options": {
    "firecrawl_api_key": "fc-xxxx",
    "db_type": "sqlite"
  }
}
```

### 运行

```bash
# 正常模式（Rich 终端 UI）
python agent.py

# 调试模式（纯文本输出，适合 IDE 内置终端）
python agent.py --debug
```

## 项目结构

```
AgentBox/
├── agent.py                     # 主入口，REPL 编排
├── keybinding.py                # prompt_toolkit 快捷键绑定
├── config.json                  # 运行时配置文件
├── config.example.json          # 配置模板
├── core/
│   ├── config.py                # AppConfig 配置模型 + ConfigManager 运行时编辑
│   ├── llm_builder.py           # LLM 工厂（自动检测 provider）
│   ├── prompt.py                # 系统提示词
│   ├── display.py               # 终端渲染（Rich / Plain 双策略）
│   ├── session.py               # 会话生命周期管理（SQLite / MongoDB）
│   ├── history.py               # 历史消息格式化与摘要
│   ├── schemas.py               # ToolResponse / ResponseCode 统一响应模型
│   ├── agent_builder.py         # Agent 构建辅助
│   ├── impl/                    # 工具实现层
│   │   ├── sheet_impl.py        # 表格操作实现
│   │   ├── fetch_impl.py        # 网络抓取与搜索实现
│   │   ├── env_impl.py          # 文件/Shell/Python 系统操作实现
│   │   └── runtime_impl.py      # 运行时工具实现（历史获取等）
│   └── tools/                   # @tool 工具定义层（LangChain tool 装饰器）
│       ├── sheet_tools.py
│       ├── fetch_tools.py
│       ├── env_tools.py
│       └── runtime_tools.py
└── data/                        # SQLite 持久化数据
```

## 可用工具

### 表格工具

| 工具 | 说明 |
|---|---|
| `tool_get_csv_excel_path` | 扫描目录下的 CSV / Excel 文件 |
| `tool_get_columns` | 获取表格表头 |
| `tool_get_columns_content` | 获取指定列的全部内容 |
| `tool_get_row_content` | 获取行内容（支持范围查询与排序） |
| `tool_count_value_in_column` | 统计列中指定值的出现次数 |
| `tool_count_data_rows` | 统计有效数据行数 |
| `tool_calculate_add` | 数值列表求和 |
| `tool_write_to_table` | 写入或追加数据到 CSV / Excel |

### 网络工具

| 工具 | 说明 |
|---|---|
| `tool_fetch_single_url_to_md` | 抓取网页转为 Markdown |
| `tool_search_online_by_query` | 在线搜索查询 |

### 系统工具

| 工具 | 说明 |
|---|---|
| `tool_get_system_plat` | 获取系统平台信息 |
| `tool_python_executor` | 执行 Python 代码并捕获输出 |
| `tool_shell_executor` | 执行 Shell 指令并捕获输出 |
| `tool_file_read_text` | 读取文本文件内容 |
| `tool_file_write_text` | 写入文本文件（覆盖/追加） |
| `tool_file_exists` | 检查文件是否存在 |
| `tool_dir_exists` | 检查目录是否存在 |
| `tool_get_working_dir` | 获取当前工作目录 |

### 运行时工具

| 工具 | 说明 |
|---|---|
| `tool_get_history` | 获取当前会话的全部历史消息 |

## 交互命令

| 命令 | 说明 |
|---|---|
| `?` | 切换指令面板显示 |
| `exit` | 退出并保存当前会话 |
| `/session` | 打开会话选择器（最近5个 + 按日期分组） |
| `/session <id>` | 直接切换到指定会话 |
| `/history` | 查看当前会话历史消息 |
| `/clear` `/new` | 新建会话（旧会话保留） |
| `/config` | 运行时查看与编辑配置（支持热重载） |

## 调试模式

使用 `--debug` 启动后，AgentBox 绕过 Rich 渲染，使用普通 `print` 输出，兼容 PyCharm、VS Code 等 IDE 内置终端。
