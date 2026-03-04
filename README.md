# ATRI — AI-Agent QQ Bot

具备 Agentic RAG、长期人设记忆与工具调用能力的 QQ 群聊智能体。通过 NapCat (OneBot v11 WebSocket) 接入 QQ，使用 OpenAI 兼容 LLM 驱动 Function Calling。

## 特性

- **时间衰减 RAG 记忆检索**：0.7 语义相似度 + 0.3 时间衰减的混合评分，越近越相关
- **长期人设画像**：每 100 条消息自动触发 LLM 总结用户画像，CAS 原子锁防并发
- **数据冷热分离**：消息先写内存缓冲区，每 5 分钟批量刷盘，降低数据库写入压力
- **3 个 Agent 工具**：记忆检索、联网搜索、代码沙盒，LLM 自主决策调用
- **安全纵深防御**：Prompt 注入隔离、工具参数审计、输出净化、沙盒网络隔离

## 架构

### MCP 工具系统

项目采用 MCP (Model Context Protocol) 架构，将工具定义与 LLM 格式解耦，支持多种模型后端：

```
agent/mcp_server.py (标准化工具定义)
    ↓
agent/mcp_tools.py (工具注册)
    ↓
agent/llm_adapter.py (模型适配器)
    ↓
agent/llm.py (统一接口)
    ↓
core/bot.py (自动执行)
```

### 模块依赖

```
app.py → core/bot.py → core/message_parser.py
                      → core/context_assembler.py
                      → core/security.py
                      → core/rate_limiter.py
                      → agent/llm.py
                      → agent/mcp_tools.py
                      → agent/{memory_search,web_search,sandbox}.py
       → memory/database.py (asyncpg, 热缓冲+冷持久化)
       → memory/embedding.py (Qwen3-Embedding)
       → memory/persona.py (CAS锁 + LLM摘要)
```

## 前置依赖

| 组件 | 说明 |
|------|------|
| Python 3.9+ | 运行环境 |
| PostgreSQL + pgvector | 向量存储与检索 |
| NapCat | OneBot v11 WebSocket 接入 QQ |
| Docker | 代码沙盒执行环境 |
| CUDA (可选) | 加速本地 Embedding 模型 |

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 复制 `config.copy.json` 为 `config.json` 并填写配置：
```jsonc
{
    "bot_qq": "机器人QQ号",
    "napcat": {
        "ws_uri": "ws://127.0.0.1:3001",
        "access_token": "你的Token"
    },
    "database": {
        "host": "localhost",
        "port": 5432,
        "database": "postgres",
        "user": "postgres",
        "password": "密码"
    },
    "llm": {
        "api_base": "https://api.deepseek.com/v1",
        "api_key": "你的API Key",
        "model": "deepseek-chat",
        "adapter": "openai"  // 模型适配器: openai 或 native
    },
    "embedding": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "device": "cuda"
    },
    "group_whitelist": [],
    "admins": [],
    "wake_words": ["ATRI", "亚托莉"],
    "system_prompt": "你的角色设定 Prompt"
}
```

3. 确保 PostgreSQL 已启动并安装 pgvector 扩展，启动：
```bash
python app.py
```
数据库建表在启动时自动执行。

4. 安装docker

## 配置说明

| 字段 | 作用 |
|------|------|
| `bot_qq` | 机器人自身 QQ 号，用于自身消息拦截防死循环 |
| `group_whitelist` | 群聊白名单，为空则不过滤，填入群号后仅处理白名单群 |
| `admins` | 管理员 QQ 号列表，沙盒工具仅管理员可用 |
| `wake_words` | 唤醒词列表，消息包含任一词即触发回复 |
| `system_prompt` | 角色设定 Prompt，注入 system message |
| `llm.adapter` | 模型适配器：`openai` (DeepSeek/GPT/Claude) 或 `native` (本地 Qwen/GLM) |

## 触发条件

机器人在以下情况回复：
- 被 @
- 被回复（仅当回复的是机器人自己的消息时触发）
- 消息包含唤醒词
- 0.2% 概率随机插嘴

## 限流机制

| 维度 | 阈值 | 窗口 |
|------|------|------|
| LLM 调用 | 每用户 6 次 | 60 秒 |
| 联网搜索 | 每用户 3 次 | 60 秒 |
| 代码沙盒 | 每用户 2 次 | 300 秒 |
| 全局熔断 | 10 次 API 失败 | 60 秒冷却 |

## 添加新工具

编辑 `agent/mcp_tools.py`：

```python
# 定义处理函数
async def my_tool_handler(param1: str, param2: int = 10, **ctx):
    group_id = ctx.get("group_id")  # 自动注入的上下文
    user_id = ctx.get("user_id")
    # 你的逻辑
    return "结果"

# 注册到 MCP Server
server.register(MCPTool(
    name="my_tool",
    description="工具描述",
    parameters={
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "参数1"},
            "param2": {"type": "integer", "description": "参数2"}
        },
        "required": ["param1"]
    },
    handler=my_tool_handler
))
```
