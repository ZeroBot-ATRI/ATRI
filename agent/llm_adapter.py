"""
LLM 适配器：支持不同模型的工具调用格式
"""
import logging
import json
import re
from typing import Protocol
from core.llm_logger import save_simple_chat, save_tool_chat

logger = logging.getLogger("atri.agent.llm_adapter")


class LLMAdapter(Protocol):
    """LLM 适配器接口"""
    async def chat_with_tools(self, messages: list, tools: list, tool_executor) -> str:
        ...


class OpenAIAdapter:
    """OpenAI 兼容格式（DeepSeek/GPT/Claude 等）"""

    def __init__(self, client, max_rounds=3):
        self.client = client
        self.max_rounds = max_rounds

    async def chat_with_tools(self, messages: list, tools: list, tool_executor) -> str:
        current_messages = list(messages)
        log_rounds = []

        try:
            for round_idx in range(self.max_rounds):
                # 最后一轮不传 tools，防止模型继续调用
                payload = {
                    "model": self.client.model,
                    "messages": current_messages,
                    "max_tokens": 16384,
                }
                if round_idx < self.max_rounds - 1:
                    payload["tools"] = tools

                resp = await self.client._post("/chat/completions", payload)
                data = resp.json()
                round_log = {"request": payload, "response": data}

                choice = data["choices"][0]
                msg = choice["message"]
                content = msg.get("content", "")

                # 检测并清理 XML 格式的工具调用（防止 LLM 输出错误格式）
                if "<tool_call>" in content:
                    logger.warning(f"检测到 XML 格式工具调用，已清理: {content[:100]}")
                    content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
                    msg["content"] = content

                current_messages.append(msg)

                if not msg.get("tool_calls"):
                    log_rounds.append(round_log)
                    return content or ""

                tool_results = []
                for tc in msg["tool_calls"]:
                    fn_name = tc["function"]["name"]
                    fn_args = json.loads(tc["function"]["arguments"])
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

            return (msg.get("content") or "思考了太久，脑子转不动了...").strip()
        finally:
            save_tool_chat(log_rounds)


class QwenAdapter:
    """Qwen XML 格式适配器"""

    def __init__(self, client, max_rounds=7):
        self.client = client
        self.max_rounds = max_rounds

    def _parse_xml_tool_calls(self, text: str) -> list:
        """解析 Qwen 的 XML 格式工具调用"""
        tool_calls = []
        pattern = r'<tool_call>\s*<function=(\w+)>\s*<parameter=(\w+)>\s*(.*?)\s*</parameter>\s*</function>\s*</tool_call>'
        matches = re.finditer(pattern, text, re.DOTALL)

        for match in matches:
            fn_name = match.group(1)
            param_name = match.group(2)
            param_value = match.group(3).strip()
            tool_calls.append({
                "name": fn_name,
                "arguments": {param_name: param_value}
            })

        return tool_calls

    async def chat_with_tools(self, messages: list, tools: list, tool_executor) -> str:
        current_messages = list(messages)
        log_rounds = []

        try:
            for _ in range(self.max_rounds):
                payload = {
                    "model": self.client.model,
                    "messages": current_messages,
                    "tools": tools,
                    "max_tokens": 16384,
                }
                resp = await self.client._post("/chat/completions", payload)
                data = resp.json()
                round_log = {"request": payload, "response": data}

                choice = data["choices"][0]
                msg = choice["message"]
                content = msg.get("content", "")

                # 检查是否包含 XML 工具调用
                tool_calls = self._parse_xml_tool_calls(content)

                if not tool_calls:
                    # 移除 XML 标签后返回
                    clean_content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
                    log_rounds.append(round_log)
                    return clean_content or content

                current_messages.append(msg)
                tool_results = []

                for tc in tool_calls:
                    fn_name = tc["name"]
                    fn_args = tc["arguments"]
                    result = await tool_executor(fn_name, fn_args)
                    tool_results.append({
                        "name": fn_name,
                        "arguments": fn_args,
                        "result": str(result),
                    })
                    current_messages.append({
                        "role": "tool",
                        "name": fn_name,
                        "content": str(result),
                    })

                round_log["tool_results"] = tool_results
                log_rounds.append(round_log)

            return (msg.get("content") or "思考了太久，脑子转不动了...").strip()
        finally:
            save_tool_chat(log_rounds)


class NativeAdapter:
    """原生格式（GLM 等本地模型）"""

    def __init__(self, client, max_rounds=7):
        self.client = client
        self.max_rounds = max_rounds

    async def chat_with_tools(self, messages: list, tools: list, tool_executor) -> str:
        current_messages = list(messages)
        log_rounds = []

        try:
            for _ in range(self.max_rounds):
                payload = {
                    "model": self.client.model,
                    "messages": current_messages,
                    "tools": tools,
                    "max_tokens": 16384,
                }
                resp = await self.client._post("/chat/completions", payload)
                data = resp.json()
                round_log = {"request": payload, "response": data}

                choice = data["choices"][0]
                msg = choice["message"]
                current_messages.append(msg)

                # Qwen 格式：tool_calls 可能在 content 中
                tool_calls = msg.get("tool_calls")
                if not tool_calls:
                    log_rounds.append(round_log)
                    return (msg.get("content") or "").strip()

                tool_results = []
                for tc in tool_calls:
                    fn_name = tc.get("function", {}).get("name") or tc.get("name")
                    fn_args_str = tc.get("function", {}).get("arguments") or tc.get("arguments", "{}")
                    fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str

                    result = await tool_executor(fn_name, fn_args)
                    tool_results.append({
                        "name": fn_name,
                        "arguments": fn_args,
                        "result": str(result),
                    })
                    current_messages.append({
                        "role": "tool",
                        "name": fn_name,
                        "content": str(result),
                    })

                round_log["tool_results"] = tool_results
                log_rounds.append(round_log)

            return (msg.get("content") or "思考了太久，脑子转不动了...").strip()
        finally:
            save_tool_chat(log_rounds)
