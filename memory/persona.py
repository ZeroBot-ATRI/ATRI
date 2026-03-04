import logging
import asyncio

logger = logging.getLogger("atri.persona")


class PersonaManager:
    """用户人设管理：CAS 原子锁抢占、异步 LLM 总结更新"""

    def __init__(self, db, llm_client=None):
        self.db = db
        self.llm = llm_client

    async def on_new_message(self, group_id: str, user_id: str):
        """消息入库后调用：自增计数 + 检查是否触发人设更新"""
        if not self.llm:
            return  # LLM 未配置，跳过人设更新

        await self.db.increment_message_count(group_id, user_id)

        # 尝试抢锁
        lock_row = await self.db.try_acquire_persona_lock(group_id, user_id)
        if lock_row:
            # 抢锁成功，投递异步更新任务
            asyncio.create_task(self._update_persona(group_id, user_id))

    async def _update_persona(self, group_id: str, user_id: str):
        """拉取最近消息，调用 LLM 总结画像，更新数据库"""
        try:
            logger.info(f"开始更新人设 {user_id}@{group_id}")

            rows = await self.db.get_recent_user_messages(group_id, user_id, limit=50)
            if not rows:
                logger.warning(f"未找到用户消息 {user_id}@{group_id}")
                return

            logger.debug(f"获取到 {len(rows)} 条用户消息")

            # 拼接消息文本（仅内容，去掉时间戳减少 token）
            messages_text = "\n".join(r['content'] for r in reversed(rows))

            # 获取当前画像
            current_persona = await self.db.get_persona(group_id, user_id)
            logger.debug(f"当前画像: {current_persona[:50]}...")

            # 精简 prompt
            prompt = (
                f"用户最近发言：\n{messages_text}\n\n"
                f"当前画像：{current_persona}\n\n"
                "根据发言更新画像（性格、兴趣、风格），100字内。"
            )

            logger.info(f"调用 LLM 生成人设，prompt 长度: {len(prompt)}")
            new_persona = await self.llm.simple_chat(prompt, timeout=120)

            if new_persona:
                logger.debug(f"LLM 返回人设: {new_persona[:100]}...")
                await self.db.update_persona(group_id, user_id, new_persona)
                logger.info(f"用户 {user_id}@{group_id} 画像已更新")
            else:
                logger.warning(f"LLM 返回空人设 {user_id}@{group_id}")

        except Exception as e:
            logger.error(f"人设更新失败 {user_id}@{group_id}: {type(e).__name__}: {e}")
            logger.info(f"人设更新失败，保持现有画像不变")
        finally:
            await self.db.release_persona_lock(group_id, user_id)
