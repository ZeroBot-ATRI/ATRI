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
        await self.db.increment_message_count(group_id, user_id)

        # 尝试抢锁
        lock_row = await self.db.try_acquire_persona_lock(group_id, user_id)
        if lock_row:
            # 抢锁成功，投递异步更新任务
            asyncio.create_task(self._update_persona(group_id, user_id))

    async def _update_persona(self, group_id: str, user_id: str):
        """拉取最近消息，调用 LLM 总结画像，更新数据库"""
        try:
            rows = await self.db.get_recent_user_messages(group_id, user_id, limit=50)
            if not rows:
                return

            # 拼接消息文本
            messages_text = "\n".join(
                f"[{r['timestamp']}] {r['content']}" for r in reversed(rows)
            )

            # 获取当前画像
            current_persona = await self.db.get_persona(group_id, user_id)

            # 调用 LLM 总结
            prompt = (
                f"以下是QQ用户 {user_id} 在群聊中的最近50条发言记录：\n"
                f"<messages>\n{messages_text}\n</messages>\n\n"
                f"该用户当前的画像是：\n{current_persona}\n\n"
                "请根据以上发言记录，更新并凝练该用户的人物画像。"
                "包括：性格特点、兴趣爱好、说话风格、常聊话题等。"
                "用简洁的第三人称描述，控制在200字以内。"
            )

            new_persona = await self.llm.simple_chat(prompt)
            if new_persona:
                await self.db.update_persona(group_id, user_id, new_persona)
                logger.info(f"用户 {user_id}@{group_id} 画像已更新")

        except Exception as e:
            logger.error(f"人设更新失败 {user_id}@{group_id}: {e}")
        finally:
            await self.db.release_persona_lock(group_id, user_id)
