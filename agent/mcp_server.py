"""
MCP-inspired Tool Server: 模型无关的工具定义与执行层
将工具定义与 LLM 格式解耦，支持多种模型后端
"""
import logging
from typing import Callable, Any
from dataclasses import dataclass

logger = logging.getLogger("atri.agent.mcp_server")


@dataclass
class MCPTool:
    """标准化工具定义"""
    name: str
    description: str
    parameters: dict  # JSON Schema
    handler: Callable  # async callable


class MCPServer:
    """工具服务器：注册、查询、执行工具"""

    def __init__(self):
        self.tools: dict[str, MCPTool] = {}

    def register(self, tool: MCPTool):
        """注册工具"""
        self.tools[tool.name] = tool
        logger.info(f"已注册工具: {tool.name}")

    def list_tools(self) -> list[MCPTool]:
        """列出所有工具"""
        return list(self.tools.values())

    async def execute(self, name: str, arguments: dict) -> Any:
        """执行工具调用"""
        if name not in self.tools:
            raise ValueError(f"未知工具: {name}")

        tool = self.tools[name]
        logger.info(f"执行工具: {name}({arguments})")
        return await tool.handler(**arguments)

    def to_openai_format(self) -> list[dict]:
        """转换为 OpenAI Function Calling 格式"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            }
            for t in self.tools.values()
        ]

    def to_native_format(self) -> list[dict]:
        """转换为原生格式（Qwen/GLM 等本地模型）"""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self.tools.values()
        ]
