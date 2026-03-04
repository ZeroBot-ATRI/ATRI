# MCP 工具系统迁移指南

## 概述

项目已重构为 MCP (Model Context Protocol) 架构，将工具定义与 LLM 格式解耦，支持多种模型后端。

## 架构变化

### 旧架构
```
agent/tools.py (OpenAI 格式硬编码)
    ↓
agent/llm.py (OpenAI Function Calling)
    ↓
core/bot.py (手动分发工具调用)
```

### 新架构 (MCP)
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

## 配置说明

在 `config.json` 的 `llm` 部分添加 `adapter` 字段：

### OpenAI 兼容模型 (DeepSeek/GPT/Claude)
```json
{
  "llm": {
    "api_base": "https://api.deepseek.com/v1",
    "api_key": "sk-xxx",
    "model": "deepseek-chat",
    "adapter": "openai"
  }
}
```

### 本地模型 (Qwen/GLM)
```json
{
  "llm": {
    "api_base": "http://localhost:8000/v1",
    "api_key": "EMPTY",
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "adapter": "native"
  }
}
```

## 支持的适配器

- `openai`: OpenAI Function Calling 格式 (默认)
  - 支持: DeepSeek, GPT-4, Claude, 通义千问在线版

- `native`: 原生工具调用格式
  - 支持: Qwen 本地部署, ChatGLM, 其他本地模型

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

## 优势

1. **模型无关**: 切换模型只需改配置，无需修改代码
2. **统一接口**: 所有工具通过 MCP Server 统一管理
3. **易于扩展**: 添加新工具只需注册，无需修改调用逻辑
4. **自动适配**: 适配器自动转换为目标模型的格式

## 兼容性

- 保留所有原有工具: `query_chat_memory`, `run_search`, `run_sandbox`
- 保留所有安全审计: Prompt Injection 防护、限流、权限检查
- 数据库、日志系统无变化
