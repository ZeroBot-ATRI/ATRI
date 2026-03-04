"""
标准 MCP Server 实现：使用官方 mcp SDK 的类型定义
进程内运行，不使用 stdio transport（工具都是本地异步函数）
"""
import logging
from typing import Callable, Any
from mcp.types import Tool, TextContent

logger = logging.getLogger("atri.agent.mcp_server")


class MCPServer:
    """MCP Server：使用标准 MCP Tool 定义，进程内执行"""

    def __init__(self):
        self.tools: dict[str, tuple[Tool, Callable]] = {}

    def register(self, tool: Tool, handler: Callable):
        """注册工具（标准 MCP Tool + 处理函数）"""
        self.tools[tool.name] = (tool, handler)
        logger.info(f"已注册工具: {tool.name}")

    def list_tools(self) -> list[Tool]:
        """列出所有工具（标准 MCP Tool 对象）"""
        return [t for t, _ in self.tools.values()]

    async def execute(self, name: str, arguments: dict) -> Any:
        """执行工具调用"""
        if name not in self.tools:
            raise ValueError(f"未知工具: {name}")

        _, handler = self.tools[name]
        logger.info(f"执行工具: {name}({arguments})")
        return await handler(**arguments)

    def to_openai_format(self) -> list[dict]:
        """转换为 OpenAI Function Calling 格式"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                }
            }
            for tool, _ in self.tools.values()
        ]

    def to_native_format(self) -> list[dict]:
        """转换为原生格式（Qwen/GLM 等本地模型）"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            }
            for tool, _ in self.tools.values()
        ]
