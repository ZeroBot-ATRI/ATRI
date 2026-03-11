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

                # 清理思考内容（以 </think> 为分隔符）
                if "</think>" in content:
                    parts = content.split("</think>", 1)
                    thinking = parts[0].strip()
                    if thinking:
                        logger.debug(f"[Thinking] {thinking[:200]}...")
                    content = parts[1].strip() if len(parts) > 1 else ""
                    msg["content"] = content

                # 再次检查：如果 content 为空但有 tool_calls，说明模型只想调用工具
                # 如果 content 不为空但包含 XML 工具调用，说明格式错误，需要清理
                if content and "<tool_call>" in content:
                    logger.warning(f"检测到 XML 格式工具调用泄露到回复中，已清理: {content[:100]}")
                    content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
                    msg["content"] = content

                current_messages.append(msg)

                if not msg.get("tool_calls"):
                    log_rounds.append(round_log)
                    # 最终返回前再次检查并清理 </think> 标签
                    if "</think>" in content:
                        parts = content.split("</think>", 1)
                        content = parts[1].strip() if len(parts) > 1 else ""
                        logger.warning(f"最终回复中仍包含 </think> 标签，已清理")
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

    def __init__(self, client, max_rounds=3):
        self.client = client
        self.max_rounds = max_rounds

    def _parse_xml_tool_calls(self, text: str) -> list:
        """解析 Qwen 的 XML 格式工具调用"""
        tool_calls = []

        # 尝试多种正则模式
        patterns = [
            # 标准格式
            r'<tool_call>\s*<function=(\w+)>\s*<parameter=(\w+)>\s*(.*?)\s*</parameter>\s*</function>\s*</tool_call>',
            # 宽松格式（允许换行）
            r'<tool_call>[\s\n]*<function=(\w+)>[\s\n]*<parameter=(\w+)>[\s\n]*(.*?)[\s\n]*</parameter>[\s\n]*</function>[\s\n]*</tool_call>',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                fn_name = match.group(1)
                param_name = match.group(2)
                param_value = match.group(3).strip()
                logger.info(f"解析到工具调用: {fn_name}({param_name}={param_value[:50]}...)")
                tool_calls.append({
                    "name": fn_name,
                    "arguments": {param_name: param_value}
                })

            if tool_calls:
                break

        if not tool_calls and "<tool_call>" in text:
            logger.warning(f"检测到 <tool_call> 但无法解析，原始文本: {text[:200]}")

        return tool_calls

    async def chat_with_tools(self, messages: list, tools: list, tool_executor) -> str:
        current_messages = list(messages)
        log_rounds = []
        final_content = ""  # 保存最终清理后的内容

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

                # 首先清理思考内容（以 </think> 为分隔符）
                if "</think>" in content:
                    parts = content.split("</think>", 1)
                    thinking = parts[0].strip()
                    if thinking:
                        logger.debug(f"[Thinking] {thinking[:200]}...")
                    content = parts[1].strip() if len(parts) > 1 else ""
                    logger.info(f"清理 </think> 标签: 原始长度={len(msg.get('content', ''))}, 清理后长度={len(content)}")

                # 保存清理后的内容
                final_content = content

                # 检查是否包含 XML 工具调用
                tool_calls = self._parse_xml_tool_calls(content)

                if not tool_calls:
                    # 移除 XML 标签后返回
                    clean_content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
                    # 最终检查：确保没有 </think> 标签泄露
                    if "</think>" in clean_content:
                        logger.error(f"!!! 最终回复中仍包含 </think> 标签！")
                        parts = clean_content.split("</think>", 1)
                        clean_content = parts[1].strip() if len(parts) > 1 else ""
                    log_rounds.append(round_log)
                    logger.info(f"QwenAdapter 返回最终回复，长度={len(clean_content)}")
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

            # 循环结束，最终检查并返回
            # 先清理 XML 工具调用标签
            final_content = re.sub(r'<tool_call>.*?</tool_call>', '', final_content, flags=re.DOTALL).strip()
            # 再清理 </think> 标签
            if "</think>" in final_content:
                logger.error(f"!!! QwenAdapter 循环结束时仍包含 </think> 标签！")
                parts = final_content.split("</think>", 1)
                final_content = parts[1].strip() if len(parts) > 1 else ""
            logger.info(f"QwenAdapter 循环结束，返回最终回复，长度={len(final_content)}")
            return final_content or "思考了太久，脑子转不动了..."
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
        final_content = ""  # 保存最终的清理后内容

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

                # 清理思考内容（以 </think> 为分隔符）
                content = msg.get("content", "")
                original_content = content  # 保存原始内容用于调试
                if "</think>" in content:
                    parts = content.split("</think>", 1)
                    thinking = parts[0].strip()
                    if thinking:
                        logger.debug(f"[Thinking] {thinking[:200]}...")
                    content = parts[1].strip() if len(parts) > 1 else ""
                    msg["content"] = content
                    logger.info(f"清理 </think> 标签: 原始长度={len(original_content)}, 清理后长度={len(content)}")

                # 保存清理后的内容
                final_content = content

                current_messages.append(msg)

                # Qwen 格式：检查是否有标准 tool_calls 或 XML 格式工具调用
                tool_calls = msg.get("tool_calls")

                # 如果没有标准格式的 tool_calls，尝试从 content 中解析 XML 格式
                if not tool_calls and content and "<tool_call>" in content:
                    logger.info("检测到 XML 格式工具调用，尝试解析...")
                    tool_calls = self._parse_xml_tool_calls(content)
                    if tool_calls:
                        # 清理 content 中的工具调用标签
                        content = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
                        msg["content"] = content
                        current_messages[-1]["content"] = content
                        final_content = content

                if not tool_calls:
                    log_rounds.append(round_log)
                    # 最终返回前再次检查并清理 </think> 标签
                    if "</think>" in final_content:
                        logger.error(f"!!! 最终回复中仍包含 </think> 标签！原始内容: {final_content[:300]}")
                        parts = final_content.split("</think>", 1)
                        final_content = parts[1].strip() if len(parts) > 1 else ""
                        logger.warning(f"已清理，新内容: {final_content[:200]}")
                    logger.info(f"返回最终回复，长度={len(final_content)}")
                    return final_content

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

            # 循环结束，最终返回前再次检查并清理 </think> 标签
            if "</think>" in final_content:
                logger.error(f"!!! 循环结束时最终回复中仍包含 </think> 标签！原始内容: {final_content[:300]}")
                parts = final_content.split("</think>", 1)
                final_content = parts[1].strip() if len(parts) > 1 else ""
                logger.warning(f"已清理，新内容: {final_content[:200]}")
            logger.info(f"循环结束，返回最终回复，长度={len(final_content)}")
            return final_content or "思考了太久，脑子转不动了..."
        finally:
            save_tool_chat(log_rounds)
