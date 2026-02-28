import logging
import asyncio
import tempfile
import os
from functools import partial

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
            "has_reply": bool,
            "reply_to_bot": bool,  # 仅当引用的是机器人消息时为 True
            "is_at_bot": bool,
            "at_target": str | None,
        }
        """
        parts = []
        image_urls = []
        has_reply = False
        reply_to_bot = False
        is_at_bot = False
        at_target = None

        for seg in message_segments:
            seg_type = seg.get("type", "")
            data = seg.get("data", {})

            if seg_type == "reply":
                reply_text, replied_author = await self._handle_reply(data)
                if reply_text:
                    parts.insert(0, reply_text)
                    has_reply = True
                    if self.bot_qq and replied_author is not None:
                        reply_to_bot = (str(replied_author) == self.bot_qq)

            elif seg_type == "at":
                at_qq = data.get("qq", "")
                at_target = at_qq
                is_at_bot = True

            # [图像-暂时禁用] 图片仅保留占位，不做 OCR/BLIP
            elif seg_type == "image":
                parts.append("[图片]")
                # img_result = await self._handle_image(data)
                # if img_result["text"]:
                #     parts.append(img_result["text"])
                # if img_result.get("url"):
                #     image_urls.append(img_result["url"])

            elif seg_type == "text":
                text = data.get("text", "").strip()
                if text:
                    parts.append(text)

        return {
            "content": " ".join(parts),
            "image_urls": image_urls,
            "has_reply": has_reply,
            "reply_to_bot": reply_to_bot,
            "is_at_bot": is_at_bot,
            "at_target": at_target,
        }

    async def _handle_reply(self, data: dict) -> tuple[str, str | None]:
        """处理引用消息，查询原文并拼接。返回 (拼接文本, 被回复消息的作者 user_id 或 None)"""
        msg_id = data.get("id", "")
        if not msg_id:
            return "", None
        row = await self.db.get_message_by_id(str(msg_id))
        if row:
            original = row["content"][:100]
            author = row["user_id"]
            author_name = (row.get("user_name") or "").strip()
            who = f"{author_name}(QQ:{author})" if author_name else f"QQ:{author}"
            return f'[回复 {who} 的消息："{original}"]', author
        return "[回复了一条未记录的消息]", None

    # [图像-暂时禁用] 本地 OCR/BLIP 图片处理
    # async def _handle_image(self, data: dict) -> dict:
    #     """处理图片消息，根据多模态配置决定处理方式"""
    #     url = data.get("url", "")
    #     if not url:
    #         return {"text": "", "url": None}
    #     if self.multimodal:
    #         return {"text": "[图片]", "url": url}
    #     if not self.vision:
    #         return {"text": "[图片]", "url": url}
    #     try:
    #         image_path = await self._download_image(url)
    #         text_parts = []
    #         loop = asyncio.get_event_loop()
    #         ocr_text = await loop.run_in_executor(
    #             None, partial(self.vision["ocr"].extract_text, image_path))
    #         if ocr_text:
    #             text_parts.append(f'[图片中的文字："{ocr_text}"]')
    #         caption = await loop.run_in_executor(
    #             None, partial(self.vision["captioner"].describe, image_path))
    #         if caption:
    #             text_parts.append(f'[图片场景："{caption}"]')
    #         os.remove(image_path)
    #         return {"text": " ".join(text_parts) if text_parts else "[图片]", "url": url}
    #     except Exception as e:
    #         logger.warning(f"图片处理失败: {e}")
    #         return {"text": "[图片]", "url": url}
    #
    # async def _download_image(self, url: str) -> str:
    #     """下载图片到临时目录，返回本地路径"""
    #     async with httpx.AsyncClient(timeout=10) as client:
    #         resp = await client.get(url)
    #         resp.raise_for_status()
    #     suffix = ".png" if "png" in url.lower() else ".jpg"
    #     fd, path = tempfile.mkstemp(suffix=suffix)
    #     with os.fdopen(fd, "wb") as f:
    #         f.write(resp.content)
    #     return path
