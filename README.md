# AgentBox

基于 LangChain / LangGraph 的智能文档处理 Agent，支持 CSV 和 Excel 文件的交互式查询与分析。


## 项目结构

```
AgentBox/
├── agent.py                    # 主入口：Agent 初始化、会话管理与交互循环
├── core/
│   ├── __init__.py             # 包标记
│   ├── sheet_tools.py          # 表格处理工具注册
│   ├── sheet_utils.py          # 底层表格数据处理逻辑
│   ├── tools.py                # 通用工具注册
│   └── utils.py                # 通用工具底层逻辑
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
pip install langchain langchain-openai langchain-deepseek langchain-community langchain-classic langgraph
pip install pandas pymongo python-dotenv
pip install questionary rich openpyxl
```
注意： 需要按照该[issue](https://github.com/langchain-ai/langchain/issues/37174)来改动langchain-deepseek(v1.0.1)的`_get_request_payload`。目前该包会导致`Missing reasoning_content`
`问题。
### 配置环境变量

编辑 `.env` 文件，填入以下配置：

```env
# LLM API（DeepSeek）
CHAT_MODEL=your_model_name_here
BASE_URL=your_base_url_here
API_KEY=your_api_key_here

# 总结模型（OpenAI 兼容，用于会话历史总结）
SUMMARY_MODEL=your_summary_model_here

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
# 普通模式
python agent.py

# 调试模式（使用普通 print 输出，用于在IDE中调试）
python agent.py --debug
```

启动后进入交互式命令行，输入 `exit` 退出。

**会话管理：**
- 输入 `/session` 列出并选择历史会话进行恢复
- 输入 `/session <session_id>` 直接切换到指定会话（格式：`yyyymmdd-xxxxxxxx`），并且会触发消息总结
- 退出时自动保存当前会话，下次可通过 `/session` 恢复

## 使用示例

```
> 帮我找一下 data 目录下有哪些文件

> 读取 data/sales.csv 的表头

> 统计 sales.csv 中 "平台" 列里 "淘宝" 出现的次数

> 查看 sales.csv 的第 1 到 10 行，按销售额降序排列

> 获取当前会话的历史消息

> /session 20260507-db733ff4

> exit
```