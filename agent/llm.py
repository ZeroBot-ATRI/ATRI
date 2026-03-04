import logging
import json
import asyncio
import httpx
from openai import OpenAI
from core.llm_logger import save_simple_chat, save_tool_chat
from agent.llm_adapter import OpenAIAdapter, NativeAdapter, QwenAdapter

logger = logging.getLogger("atri.agent.llm")

TIMEOUT = 60  # API 超时秒数（思考模型需要更长时间）


class LLMClient:
    """OpenAI 兼容 LLM 客户端，支持 MCP 工具调用"""

    def __init__(self, config: dict):
        self.api_base = config["api_base"].rstrip("/")
        self.api_key = config["api_key"]
        self.model = config["model"]
        self.multimodal = config.get("multimodal", False)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # OpenAI SDK 客户端（用于 simple_chat，使用同步版本）
        self.openai_client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            timeout=120.0
        )

        # 根据配置选择适配器
        adapter_type = config.get("adapter", "openai")
        if adapter_type == "qwen":
            self.adapter = QwenAdapter(self)
        elif adapter_type == "native":
            self.adapter = NativeAdapter(self)
        else:
            self.adapter = OpenAIAdapter(self)

    async def _post(self, endpoint: str, payload: dict):
        """统一 HTTP POST 请求"""
        return await self._post_with_timeout(endpoint, payload, TIMEOUT)

    async def _post_with_timeout(self, endpoint: str, payload: dict, timeout: int):
        """统一 HTTP POST 请求（可自定义超时）"""
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        f"{self.api_base}{endpoint}",
                        headers=self.headers,
                        json=payload,
                    )
                    if resp.status_code != 200:
                        logger.error(f"LLM API 返回 {resp.status_code}: {resp.text}")
                        resp.raise_for_status()
                    return resp
            except (httpx.RemoteProtocolError, httpx.ReadError) as e:
                logger.warning(f"连接断开 (尝试 {attempt+1}/3): {e}")
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def simple_chat(self, prompt: str, disable_thinking: bool = False, timeout: int = None) -> str:
        """简单对话（无工具），使用同步 OpenAI SDK + asyncio.to_thread"""
        def _sync_call():
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=16384,
                temperature=1.0,
                stream=True
            )
            # 收集流式响应
            content = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
            return content.strip()

        try:
            content = await asyncio.to_thread(_sync_call)
            return content
        except Exception as e:
            logger.error(f"OpenAI SDK 调用失败: {e}")
            raise

    async def chat_with_tools(self, messages: list, tools: list,
                              tool_executor) -> str:
        """
        带工具调用的多轮对话循环（委托给适配器）
        tool_executor: async callable(name, args) -> str
        返回最终文本回复
        """
        return await self.adapter.chat_with_tools(messages, tools, tool_executor)
