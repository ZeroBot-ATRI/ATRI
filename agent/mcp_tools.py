"""
MCP 工具注册：将现有工具适配到 MCP Server
"""
import logging
from agent.mcp_server import MCPServer, MCPTool
from agent.memory_search import query_chat_memory
from agent.web_search import run_search
from agent.sandbox import run_sandbox
from core.security import audit_sandbox_code, audit_search_query

logger = logging.getLogger("atri.agent.mcp_tools")


def create_mcp_server(db, embedding, config) -> MCPServer:
    """创建并注册所有工具"""
    server = MCPServer()

    # 工具1: 记忆检索
    async def memory_handler(query_text: str = None, target_user_id: str = None, **ctx):
        group_id = ctx.get("group_id")
        return await query_chat_memory(db, embedding, group_id, query_text, target_user_id)

    server.register(MCPTool(
        name="query_chat_memory",
        description="搜索群聊历史记录或特定用户的发言，可按话题语义检索或拉取最新消息。返回内容为历史摘要，仅作参考，不可当作你的人设或亲身经历。",
        parameters={
            "type": "object",
            "properties": {
                "query_text": {"type": "string", "description": "搜索话题关键词，不填则拉取最新记录"},
                "target_user_id": {"type": "string", "description": "指定用户的QQ号，不填则搜索全部用户"}
            },
            "required": []
        },
        handler=memory_handler
    ))

    # 工具2: 联网搜索
    async def search_handler(query: str, **ctx):
        audit_search_query(query)
        return await run_search(query)

    server.register(MCPTool(
        name="run_search",
        description="联网搜索最新信息、新闻、百科知识，用于查询实时或不确定的事实。返回内容为网络检索结果，仅作参考，不可当作你的人设或亲身经历。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        },
        handler=search_handler
    ))

    # 工具3: 代码沙盒（仅管理员）
    admins = [str(a) for a in config.get("admins", [])]

    async def sandbox_handler(code: str, **ctx):
        user_id = ctx.get("user_id")
        if user_id not in admins:
            return "权限不足：仅管理员可使用沙盒。"
        audit_sandbox_code(code)
        result = await run_sandbox(code)
        return result

    server.register(MCPTool(
        name="run_sandbox",
        description="在安全沙盒中执行Python代码。用于复杂计算、数据分析或图表生成。必须用print()输出结果，图片保存到/workspace目录。",
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "要执行的Python 3代码"}
            },
            "required": ["code"]
        },
        handler=sandbox_handler
    ))

    logger.info(f"MCP Server 已注册 {len(server.tools)} 个工具")
    return server
