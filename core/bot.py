import logging
import json
import random
import asyncio
import websockets

from core.message_parser import MessageParser
from core.context_assembler import ContextAssembler
from core.security import (
    audit_sandbox_code, audit_search_query,
    # audit_vision_url,  # [图像-暂时禁用]
    sanitize_output,
)
from core.rate_limiter import RateLimiter
from memory.database import sanitize_user_name
from agent.memory_search import query_chat_memory
from agent.web_search import run_search
from agent.sandbox import run_sandbox
# from agent.vision import run_vision  # [图像-暂时禁用]

logger = logging.getLogger("atri.bot")


class Bot:
    """WebSocket 主循环：消息接收、处理、回复"""

    def __init__(self, config, db, embedding, llm, tool_registry,
                 persona_manager, vision_pipeline=None):
        self.config = config
        self.db = db
        self.embedding = embedding
        self.llm = llm
        self.tools = tool_registry
        self.persona = persona_manager
        self.vision_pipeline = vision_pipeline
        self.rate_limiter = RateLimiter()

        self.bot_qq = str(config["bot_qq"])
        self.wake_words = config.get("wake_words", [])
        self.admins = [str(a) for a in config.get("admins", [])]
        self.multimodal = config["llm"].get("multimodal", False)
        self.group_whitelist = [str(g) for g in config.get("group_whitelist", [])]

        self._echo_counter = 0

        self.parser = MessageParser(db, vision_pipeline, self.multimodal, self.bot_qq)
        self.assembler = ContextAssembler(
            config.get("system_prompt", ""), self.multimodal
        )

    async def run(self):
        """启动 WebSocket 连接并监听消息"""
        uri = self.config["napcat"]["ws_uri"]
        token = self.config["napcat"]["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        logger.info(f"正在连接 NapCat: {uri}")
        async with websockets.connect(uri, additional_headers=headers) as ws:
            logger.info("WebSocket 连接成功，开始监听消息...")
            async for raw in ws:
                try:
                    data = json.loads(raw)
                    await self._dispatch(data, ws)
                except Exception as e:
                    logger.error(f"消息处理异常: {e}", exc_info=True)

    async def _dispatch(self, data: dict, ws):
        """消息分发：过滤心跳、拦截自身消息、处理群聊"""
        # 过滤心跳
        if data.get("meta_event_type") == "heartbeat":
            return

        # 仅处理群聊消息
        if data.get("post_type") != "message" or data.get("message_type") != "group":
            return

        group_id = str(data.get("group_id", ""))
        user_id = str(data.get("user_id", ""))
        message_id = str(data.get("message_id", ""))
        segments = data.get("message", [])
        sender = data.get("sender", {})
        raw_name = sender.get("card") or sender.get("nickname") or ""
        user_name = sanitize_user_name(raw_name) if raw_name else ""

        if not segments or not group_id:
            return

        # 群聊白名单过滤：白名单非空时，仅处理白名单内的群
        if self.group_whitelist and group_id not in self.group_whitelist:
            return

        # 解析消息
        parsed = await self.parser.parse(segments)
        content = parsed["content"]
        if not content.strip():
            return

        # 长度 < 5 的消息不做嵌入
        embedding_vec = None
        if len(content.strip()) >= 5:
            embedding_vec = await self.embedding.get_embedding(content)

        # 自身消息拦截（最高优先级，防死循环）
        if user_id == self.bot_qq:
            return

        # 写入热存储缓冲区
        await self.db.buffer_message(
            message_id, group_id, user_id, "user", content, embedding_vec,
            user_name=user_name,
        )

        # 人设计数
        await self.persona.on_new_message(group_id, user_id)

        # 触发判定
        should_reply = self._should_reply(parsed, content)
        if not should_reply:
            return

        # 限流检查
        if self.rate_limiter.is_circuit_open():
            await self._send_group(ws, group_id, "我脑袋有点宕机，稍微歇一下再问我吧")
            return
        if not self.rate_limiter.check_llm(user_id):
            return

        # Prompt 组装
        try:
            persona_text = await self.db.get_persona(group_id, user_id)
            recent = await self.db.get_recent_messages(group_id, limit=20)
            messages = self.assembler.assemble(
                persona_text, recent, content, None  # parsed.get("image_urls") [图像-暂时禁用]
            )
        except Exception as e:
            logger.error(f"Prompt 组装失败: {e}")
            return

        # LLM 调用（带工具）
        try:
            reply = await self.llm.chat_with_tools(
                messages,
                self.tools.get_tool_definitions(),
                lambda name, args: self._execute_tool(name, args, group_id, user_id),
            )
            self.rate_limiter.record_api_success()
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            self.rate_limiter.record_api_failure()
            await self._send_group(ws, group_id, "我脑袋有点宕机，稍微歇一下再问我吧")
            return

        # 输出净化 + 发送 + 主动存储机器人回复
        if reply:
            reply = sanitize_output(reply)
            sent_msg_id = await self._send_group(ws, group_id, reply)
            bot_embedding = None
            if len(reply.strip()) >= 5:
                bot_embedding = await self.embedding.get_embedding(reply)
            await self.db.buffer_message(
                sent_msg_id, group_id, self.bot_qq, "assistant", reply, bot_embedding,
                user_name="[机器人]",
            )

    def _should_reply(self, parsed: dict, content: str) -> bool:
        """判断是否需要触发 LLM 回复"""
        # 被 @
        if parsed["is_at_bot"] and parsed["at_target"] == self.bot_qq:
            return True
        # 被回复（仅当回复的是本机器人消息时触发）
        if parsed["has_reply"] and parsed.get("reply_to_bot", False):
            return True
        # 包含唤醒词（仅在用户真实输入中匹配，排除回复前缀里的 user_id 等）
        content_for_wake = content
        if content_for_wake.startswith("[回复 QQ:") and '"]' in content_for_wake:
            content_for_wake = content_for_wake.split('"]', 1)[-1].strip()
        for word in self.wake_words:
            if word and word.lower() in content_for_wake.lower():
                return True
        # 0.2% 概率随机插嘴
        if random.random() < 0.002:
            return True
        return False

    async def _execute_tool(self, name: str, args: dict,
                            group_id: str, user_id: str) -> str:
        """工具分发器：根据名称路由到对应实现"""
        if name == "query_chat_memory":
            return await query_chat_memory(
                self.db, self.embedding, group_id,
                args.get("query_text"), args.get("target_user_id"),
            )

        elif name == "run_search":
            query = args.get("query", "")
            block = audit_search_query(query)
            if block:
                return block
            if not self.rate_limiter.check_search(user_id):
                return "搜索太频繁了，稍后再试。"
            return await run_search(query)

        elif name == "run_sandbox":
            code = args.get("code", "")
            block = audit_sandbox_code(code)
            if block:
                return block
            if not self.rate_limiter.check_sandbox(user_id):
                return "沙盒调用太频繁，请稍后再试。"
            result = await run_sandbox(code)
            self._pending_images = result.get("images", [])
            return result["stdout"]

        # [图像-暂时禁用] run_vision
        # elif name == "run_vision":
        #     url = args.get("image_url", "")
        #     mode = args.get("mode", "describe")
        #     block = audit_vision_url(url)
        #     if block:
        #         return block
        #     if not self.vision_pipeline:
        #         return "视觉模型未加载。"
        #     return await run_vision(self.vision_pipeline, url, mode)

        return f"未知工具: {name}"

    async def _send_group(self, ws, group_id: str, text: str) -> str:
        """发送群聊消息，直接读 WebSocket 等待 echo 响应获取 message_id"""
        message = text
        images = getattr(self, "_pending_images", [])
        if images:
            for b64 in images:
                message += f"\n[CQ:image,file=base64://{b64}]"
            self._pending_images = []

        self._echo_counter += 1
        echo = f"send_{self._echo_counter}"

        payload = {
            "action": "send_group_msg",
            "echo": echo,
            "params": {
                "group_id": int(group_id),
                "message": message,
            },
        }
        await ws.send(json.dumps(payload))
        logger.info(f"已发送回复到群 {group_id}")

        # 直接读 WebSocket 等待 echo 响应，其他消息异步分发
        deadline = asyncio.get_event_loop().time() + 5
        try:
            while asyncio.get_event_loop().time() < deadline:
                remaining = deadline - asyncio.get_event_loop().time()
                raw = await asyncio.wait_for(ws.recv(), timeout=max(remaining, 0.1))
                data = json.loads(raw)
                if data.get("echo") == echo:
                    return str(data.get("data", {}).get("message_id", ""))
                # 非 echo 响应的消息异步分发，不阻塞当前等待
                asyncio.create_task(self._dispatch(data, ws))
        except asyncio.TimeoutError:
            logger.warning(f"发送响应超时，echo={echo}")
        return ""
