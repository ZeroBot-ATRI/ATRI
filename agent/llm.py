import logging
import json
import httpx
from core.llm_logger import save_simple_chat, save_tool_chat

logger = logging.getLogger("atri.agent.llm")

TIMEOUT = 15  # API 超时秒数


class LLMClient:
    """OpenAI 兼容 LLM 客户端，支持 Function Calling 多轮循环"""

    def __init__(self, config: dict):
        self.api_base = config["api_base"].rstrip("/")
        self.api_key = config["api_key"]
        self.model = config["model"]
        self.multimodal = config.get("multimodal", False)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def simple_chat(self, prompt: str) -> str:
        """简单对话（无工具），用于人设总结等内部任务"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
        }
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        save_simple_chat(payload, data)
        return data["choices"][0]["message"]["content"].strip()

    async def chat_with_tools(self, messages: list, tools: list,
                              tool_executor) -> str:
        """
        带 Function Calling 的多轮对话循环
        tool_executor: async callable(name, args) -> str
        返回最终文本回复
        """
        max_rounds = 7
        current_messages = list(messages)
        log_rounds = []

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                for _ in range(max_rounds):
                    payload = {
                        "model": self.model,
                        "messages": list(current_messages),
                        "tools": tools,
                        "max_tokens": 1500,
                    }
                    resp = await client.post(
                        f"{self.api_base}/chat/completions",
                        headers=self.headers,
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    round_log = {"request": payload, "response": data}

                    choice = data["choices"][0]
                    msg = choice["message"]
                    current_messages.append(msg)

                    # 无工具调用，返回文本
                    if not msg.get("tool_calls"):
                        log_rounds.append(round_log)
                        return msg.get("content", "").strip()

                    # 执行工具调用
                    tool_results = []
                    for tc in msg["tool_calls"]:
                        fn_name = tc["function"]["name"]
                        fn_args = json.loads(tc["function"]["arguments"])
                        logger.info(f"工具调用: {fn_name}({fn_args})")

                        result = await tool_executor(fn_name, fn_args)
                        tool_results.append({
                            "tool_call_id": tc["id"],
                            "name": fn_name,
                            "arguments": fn_args,
                            "result": str(result),
                        })

                        current_messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": str(result),
                        })

                    round_log["tool_results"] = tool_results
                    log_rounds.append(round_log)

            # 超过最大轮次，取最后一条文本
            return msg.get("content", "思考了太久，脑子转不动了...")
        finally:
            save_tool_chat(log_rounds)
