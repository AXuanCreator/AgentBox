# AgentBox

基于 LangChain / LangGraph 的智能文档处理 Agent，支持 CSV 和 Excel 文件的交互式查询与分析。


## 项目结构

```
AgentBox/
├── agent.py                    # 主入口：Agent 初始化与交互循环
├── sheet_processing/
│   ├── __init__.py             # 包标记
│   ├── tools.py                # LangChain 工具注册（7 个工具）
│   └── utils.py                # 底层数据处理逻辑（pandas）
├── .env                        # 环境变量配置
├── .gitignore                  # Git 忽略规则
└── README.md                   # 项目说明
```

## 快速开始

### 环境要求

- Python 3.10+
- MongoDB 实例（用于会话持久化）

### 安装依赖

```bash
pip install langchain langchain-openai langchain-community langchain-classic langgraph
pip install pandas pymongo redis python-dotenv
pip install questionary rich openpyxl
```

### 配置环境变量

编辑 `.env` 文件，填入以下配置：

```env
# LLM API
CHAT_MODEL=your_model_name_here
BASE_URL=your_base_url_here
API_KEY=your_api_key_here

# MongoDB（会话记忆）
MONGO_SHORTMEMORY_URL=mongodb://user:password@host:port

# LangSmith 追踪（可选）
LANGCHAIN_TRACING_V2=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=agentbox
```

### 运行

```bash
python agent.py
```

启动后进入交互式命令行，输入 `exit` 退出。

## 使用示例

```
> 帮我找一下 data 目录下有哪些文件

> 读取 data/sales.csv 的表头

> 统计 sales.csv 中 "平台" 列里 "淘宝" 出现的次数

> 查看 sales.csv 的第 1 到 10 行，按销售额降序排列

> exit
```

## 可用工具

| 工具 | 功能 |
|------|------|
| `tool_get_csv_excel_path` | 扫描目录获取所有 CSV/Excel 文件路径 |
| `tool_get_columns` | 获取文件的列名（表头） |
| `tool_get_columns_content` | 获取指定列的全部内容 |
| `tool_count_value_in_column` | 统计指定值在列中出现的次数 |
| `tool_get_row_content` | 按行号获取数据内容 |
| `tool_count_data_rows` | 统计数据行数（不含表头） |
| `tool_calculate_add` | 对数字列表求和 |
