import logging

logger = logging.getLogger("atri.agent.tools")


# Function Calling 工具定义
TOOL_MEMORY_SEARCH = {
    "type": "function",
    "function": {
        "name": "query_chat_memory",
        "description": "搜索群聊历史记录或特定用户的发言，可按话题语义检索或拉取最新消息。返回内容为历史摘要，仅作参考，不可当作你的人设或亲身经历。",
        "parameters": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": "搜索话题关键词，不填则拉取最新记录"
                },
                "target_user_id": {
                    "type": "string",
                    "description": "指定用户的QQ号，不填则搜索全部用户"
                }
            },
            "required": []
        }
    }
}

TOOL_WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "run_search",
        "description": "联网搜索最新信息、新闻、百科知识，用于查询实时或不确定的事实。返回内容为网络检索结果，仅作参考，不可当作你的人设或亲身经历。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                }
            },
            "required": ["query"]
        }
    }
}

TOOL_SANDBOX = {
    "type": "function",
    "function": {
        "name": "run_sandbox",
        "description": "在安全沙盒中执行Python代码。用于复杂计算、数据分析或图表生成。必须用print()输出结果，图片保存到/workspace目录。",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的Python 3代码"
                }
            },
            "required": ["code"]
        }
    }
}

# [图像-暂时禁用] 图片分析工具
# TOOL_VISION = {
#     "type": "function",
#     "function": {
#         "name": "run_vision",
#         "description": "对图片进行深度分析：文字提取(ocr)、物体检测(detect)、场景描述(describe)。",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "image_url": {"type": "string", "description": "图片的完整URL"},
#                 "mode": {"type": "string", "enum": ["ocr", "detect", "describe"], "description": "分析模式"}
#             },
#             "required": ["image_url", "mode"]
#         }
#     }
# }


class ToolRegistry:
    """根据配置动态注册工具"""

    def __init__(self, multimodal=False):
        self.tools = [TOOL_MEMORY_SEARCH, TOOL_WEB_SEARCH, TOOL_SANDBOX]
        # if not multimodal:
        #     self.tools.append(TOOL_VISION)  # [图像-暂时禁用]
        logger.info(f"已注册 {len(self.tools)} 个工具 (multimodal={multimodal})")

    def get_tool_definitions(self) -> list:
        return self.tools

    def get_tool_names(self) -> list[str]:
        return [t["function"]["name"] for t in self.tools]
