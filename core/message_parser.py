import logging
import base64

import httpx

logger = logging.getLogger("atri.message_parser")


class MessageParser:
    """OneBot v11 消息段解析、引用拼接、图片理解、富文本组装"""

    def __init__(self, db, vision_pipeline=None, multimodal=False, bot_qq=None):
        self.db = db
        self.vision = vision_pipeline
        self.multimodal = multimodal
        self.bot_qq = str(bot_qq) if bot_qq else None

    async def parse(self, message_segments: list) -> dict:
        """
        解析消息段数组，返回:
        {
            "content": 拼接后的富文本,
            "image_urls": [图片URL列表，多模态模式下使用],
            "raw_image_urls": [原始图片URL列表，用于存储到数据库],
            "has_reply": bool,
            "reply_to_bot": bool,  # 仅当引用的是机器人消息时为 True
            "reply_author_name": str | None,  # 被引用消息的作者名
            "reply_image_urls": [被引用消息的图片URL列表],
            "is_at_bot": bool,
            "at_target": str | None,
        }
        """
        parts = []
        image_urls = []
        raw_image_urls = []
        has_reply = False
        reply_to_bot = False
        reply_author_name = None
        reply_image_urls = []
        is_at_bot = False
        at_target = None

        for seg in message_segments:
            seg_type = seg.get("type", "")
            data = seg.get("data", {})

            if seg_type == "reply":
                reply_text, replied_author, replied_author_name, reply_imgs = await self._handle_reply(data)
                if reply_text:
                    parts.insert(0, reply_text)
                    has_reply = True
                    reply_author_name = replied_author_name
                    reply_image_urls = reply_imgs or []
                    if self.bot_qq and replied_author is not None:
                        reply_to_bot = (str(replied_author) == self.bot_qq)

            elif seg_type == "at":
                at_qq = data.get("qq", "")
                at_target = at_qq
                is_at_bot = True

            # 多模态：下载图片并转为 base64 data URI 发送给 LLM
            elif seg_type == "image":
                parts.append("[图片]")
                url = data.get("url", "")
                if url:
                    raw_image_urls.append(url)
                    if self.multimodal:
                        data_uri = await self._download_image_as_base64(url)
                        if data_uri:
                            image_urls.append(data_uri)

            elif seg_type == "text":
                text = data.get("text", "").strip()
                if text:
                    parts.append(text)

        return {
            "content": " ".join(parts),
            "image_urls": image_urls,
            "raw_image_urls": raw_image_urls,
            "has_reply": has_reply,
            "reply_to_bot": reply_to_bot,
            "reply_author_name": reply_author_name,
            "reply_image_urls": reply_image_urls,
            "is_at_bot": is_at_bot,
            "at_target": at_target,
        }

    async def _handle_reply(self, data: dict) -> tuple[str, str | None, str | None, list]:
        """处理引用消息，查询原文并拼接。返回 (拼接文本, 被回复消息的作者 user_id, 作者名, 图片URL列表)"""
        msg_id = data.get("id", "")
        if not msg_id:
            return "", None, None, []
        row = await self.db.get_message_by_id(str(msg_id))
        if row:
            original = row["content"][:100]
            author = row["user_id"]
            author_name = (row.get("user_name") or "").strip()
            who = f"{author_name}(QQ:{author})" if author_name else f"QQ:{author}"

            # 获取图片URLs
            image_urls_str = row.get("image_urls")
            image_urls = []
            if image_urls_str:
                import json
                try:
                    image_urls = json.loads(image_urls_str)
                except:
                    pass

            return f'[回复 {who} 的消息："{original}"]', author, author_name, image_urls
        return "[回复了一条未记录的消息]", None, None, []

    async def _download_image_as_base64(self, url: str) -> str | None:
        """下载图片并转为 base64 data URI，供多模态 LLM 使用"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            content_type = resp.headers.get("content-type", "image/jpeg")
            mime = content_type.split(";")[0].strip()
            b64 = base64.b64encode(resp.content).decode()
            return f"data:{mime};base64,{b64}"
        except Exception as e:
            logger.warning(f"图片下载失败: {e}")
            return None
