import logging
import json
import random
import asyncio
import time
import websockets

from core.message_parser import MessageParser
from core.context_assembler import ContextAssembler
from core.security import sanitize_output
from core.rate_limiter import RateLimiter
from memory.database import sanitize_user_name

logger = logging.getLogger("atri.bot")


class Bot:
    """WebSocket 主循环：消息接收、处理、回复"""

    def __init__(self, config, db, embedding, llm, mcp_server,
                 persona_manager, vision_pipeline=None):
        self.config = config
        self.db = db
        self.embedding = embedding
        self.llm = llm
        self.mcp = mcp_server
        self.persona = persona_manager
        self.vision_pipeline = vision_pipeline
        self.rate_limiter = RateLimiter()

        self.bot_qq = str(config["bot_qq"])
        self.wake_words = config.get("wake_words", [])
        self.admins = [str(a) for a in config.get("admins", [])]
        self.multimodal = config["llm"].get("multimodal", False)
        self.group_whitelist = [str(g) for g in config.get("group_whitelist", [])]

        self._echo_counter = 0
        self._last_random_reply = 0.0
        self._pending_echos = {}  # echo_id -> asyncio.Future[str]

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
                    # Echo 响应路由
                    echo = data.get("echo")
                    if echo and echo in self._pending_echos:
                        future = self._pending_echos.pop(echo)
                        msg_id = str(data.get("data", {}).get("message_id", ""))
                        future.set_result(msg_id)
                    else:
                        # 并发处理普通消息
                        asyncio.create_task(self._dispatch(data, ws))
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
                persona_text, recent, content, parsed.get("image_urls")
            )
        except Exception as e:
            logger.error(f"Prompt 组装失败: {e}")
            return

        # LLM 调用（带工具）
        try:
            adapter_type = self.config["llm"].get("adapter", "openai")
            tools = self.mcp.to_native_format() if adapter_type == "native" else self.mcp.to_openai_format()

            reply = await self.llm.chat_with_tools(
                messages,
                tools,
                lambda name, args: self._execute_tool(name, args, group_id, user_id),
            )
            self.rate_limiter.record_api_success()
        except Exception as e:
            logger.error(f"LLM 调用失败: {type(e).__name__}: {e}", exc_info=True)
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
                user_name="我",
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
        # 1% 概率随机插嘴，两次间隔不低于 180 秒
        if random.random() < 0.01 and time.time() - self._last_random_reply >= 180:
            self._last_random_reply = time.time()
            return True
        return False

    async def _execute_tool(self, name: str, args: dict,
                            group_id: str, user_id: str) -> str:
        """工具执行统一入口（通过 MCP Server）"""
        # 限流检查
        if name == "run_search" and not self.rate_limiter.check_search(user_id):
            return "搜索太频繁了，稍后再试。"
        if name == "run_sandbox" and not self.rate_limiter.check_sandbox(user_id):
            return "沙盒调用太频繁，请稍后再试。"

        # 注入上下文
        args["group_id"] = group_id
        args["user_id"] = user_id

        try:
            result = await self.mcp.execute(name, args)
            # 处理沙盒图片
            if name == "run_sandbox" and isinstance(result, dict):
                self._pending_images = result.get("images", [])
                return result["stdout"]
            return result
        except Exception as e:
            logger.error(f"工具执行失败 {name}: {e}", exc_info=True)
            return f"工具执行出错: {e}"

    async def _send_group(self, ws, group_id: str, text: str) -> str:
        """发送群聊消息，通过 Future 等待 echo 响应获取 message_id"""
        message = text
        images = getattr(self, "_pending_images", [])
        if images:
            for b64 in images:
                message += f"\n[CQ:image,file=base64://{b64}]"
            self._pending_images = []

        self._echo_counter += 1
        echo = f"send_{self._echo_counter}"

        # 注册 Future 等待 echo 响应
        future = asyncio.Future()
        self._pending_echos[echo] = future

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

        # 等待 echo 响应（由主循环路由）
        try:
            msg_id = await asyncio.wait_for(future, timeout=5.0)
            return msg_id
        except asyncio.TimeoutError:
            logger.warning(f"发送响应超时，echo={echo}")
            self._pending_echos.pop(echo, None)
            return ""
