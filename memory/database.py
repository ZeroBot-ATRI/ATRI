import asyncpg
import logging
import os
import re
import asyncio
from datetime import datetime

logger = logging.getLogger("atri.database")

# 用户名校验：仅保留可识别字符并截断，避免特殊名字撑爆数据库
USER_NAME_MAX_LEN = 32
_USER_NAME_SAFE_PATTERN = re.compile(
    r"[^\s\w\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af.,!?:\-_\u2018\u2019\u201c\u201d]"
)


def sanitize_user_name(name: str, max_len: int = USER_NAME_MAX_LEN) -> str:
    """剔除无法识别的字符并截断，返回安全昵称。空或全被剔除则返回空串。"""
    if not name or not isinstance(name, str):
        return ""
    # 剔除控制字符与不可识别字符
    cleaned = _USER_NAME_SAFE_PATTERN.sub("", name)
    cleaned = "".join(c for c in cleaned if ord(c) >= 32 and ord(c) != 127 and (ord(c) < 0x7f or ord(c) > 0x9f))
    cleaned = cleaned.strip()
    return cleaned[:max_len] if cleaned else ""


class Database:
    """asyncpg 连接池封装，负责 chat_memory 和 user_personas 的 CRUD"""

    def __init__(self, config: dict):
        self.config = config
        self.pool: asyncpg.Pool | None = None
        # 热存储：内存消息缓冲区
        self._buffer: list[tuple] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(
            host=self.config["host"],
            port=self.config["port"],
            database=self.config["database"],
            user=self.config["user"],
            password=self.config["password"],
            min_size=2,
            max_size=10,
        )
        logger.info("数据库连接池已建立")
        # 启动定时刷盘任务（每5分钟）
        self._flush_task = asyncio.create_task(self._periodic_flush())

    # ---- 热存储缓冲区 ----

    async def buffer_message(self, message_id, group_id, user_id, role, content, embedding, user_name: str = ""):
        """将消息写入内存缓冲区（热存储）。user_name 应由调用方先经 sanitize_user_name 处理。"""
        async with self._buffer_lock:
            self._buffer.append((message_id, group_id, user_id, user_name, role, content, embedding))
        logger.debug(f"消息已缓冲，当前缓冲区大小: {len(self._buffer)}")

    async def _periodic_flush(self):
        """每5分钟将缓冲区消息批量持久化到数据库（冷存储）"""
        while True:
            await asyncio.sleep(300)  # 5分钟
            await self._flush_buffer()

    async def _flush_buffer(self):
        """将缓冲区中的消息批量写入数据库"""
        async with self._buffer_lock:
            if not self._buffer:
                return
            batch = list(self._buffer)
            self._buffer.clear()

        if not batch:
            return

        try:
            async with self.pool.acquire() as conn:
                await conn.executemany(
                    """INSERT INTO chat_memory (message_id, group_id, user_id, user_name, role, content, embedding)
                       VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                    batch,
                )
            logger.info(f"缓冲区刷盘完成，写入 {len(batch)} 条消息")
        except Exception as e:
            logger.error(f"缓冲区刷盘失败: {e}")
            # 写回缓冲区，避免数据丢失
            async with self._buffer_lock:
                self._buffer = batch + self._buffer

    async def init_tables(self):
        sql_path = os.path.join(os.path.dirname(__file__), "..", "sql", "init.sql")
        with open(sql_path, "r", encoding="utf-8") as f:
            sql = f.read()
        async with self.pool.acquire() as conn:
            await conn.execute(sql)
        logger.info("数据库表初始化完成")

    # ---- chat_memory CRUD ----

    async def insert_message(self, message_id, group_id, user_id, role, content, embedding, user_name: str = ""):
        """写入一条聊天记录（含向量）"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO chat_memory (message_id, group_id, user_id, user_name, role, content, embedding)
                   VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                message_id, group_id, user_id, user_name, role, content, embedding,
            )

    async def get_message_by_id(self, message_id: str):
        """根据 message_id 查询单条记录（用于引用溯源），优先查缓冲区。返回 content, user_id, user_name。"""
        async with self._buffer_lock:
            for msg in self._buffer:
                if msg[0] == message_id:
                    return {"content": msg[5], "user_id": msg[2], "user_name": msg[3] or ""}
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT content, user_id, user_name FROM chat_memory WHERE message_id = $1",
                message_id,
            )
            if row:
                return {"content": row["content"], "user_id": row["user_id"], "user_name": row.get("user_name") or ""}
            return None

    async def get_recent_messages(self, group_id: str, limit: int = 20):
        """获取群聊最近 N 条记录（合并缓冲区 + 数据库），每条含 user_id, user_name, role, content, timestamp。"""
        # 从缓冲区取出本群消息 (message_id, group_id, user_id, user_name, role, content, embedding)
        buffered = []
        async with self._buffer_lock:
            for msg in self._buffer:
                if msg[1] == group_id:
                    buffered.append({
                        "user_id": msg[2], "user_name": msg[3] or "",
                        "role": msg[4], "content": msg[5],
                        "timestamp": datetime.now(),
                    })

        # 从数据库补齐
        db_limit = max(limit - len(buffered), 0)
        db_rows = []
        if db_limit > 0:
            async with self.pool.acquire() as conn:
                db_rows = await conn.fetch(
                    """SELECT user_id, COALESCE(user_name, '') AS user_name, role, content, timestamp
                       FROM chat_memory WHERE group_id = $1
                       ORDER BY timestamp DESC LIMIT $2""",
                    group_id, db_limit,
                )

        # 合并：缓冲区消息在前（更新），DB 消息在后
        return buffered[::-1] + list(db_rows) if buffered else list(db_rows)

    async def search_memory(self, group_id, embedding_vec, half_life_days=1.0,
                            target_user_id=None, limit=10):
        """时间半衰期加权 RAG 检索"""
        base_sql = """
            SELECT user_id, COALESCE(user_name, '') AS user_name, content, timestamp,
                (0.7 * (1 - (embedding <=> $1))) +
                (0.3 * EXP(-EXTRACT(EPOCH FROM (NOW() - timestamp)) / ($2 * 86400.0)))
                AS score
            FROM chat_memory
            WHERE group_id = $3 AND embedding IS NOT NULL
        """
        params = [embedding_vec, half_life_days, group_id]
        if target_user_id:
            base_sql += " AND user_id = $4"
            params.append(target_user_id)
            limit_param = "$5"
        else:
            limit_param = "$4"
        base_sql += f" ORDER BY score DESC LIMIT {limit_param}"
        params.append(limit)

        async with self.pool.acquire() as conn:
            return await conn.fetch(base_sql, *params)

    # ---- user_personas CRUD ----

    async def get_persona(self, group_id: str, user_id: str) -> str:
        """获取用户画像文本"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT persona_text FROM user_personas WHERE group_id = $1 AND user_id = $2",
                group_id, user_id,
            )
            return row["persona_text"] if row else "该用户暂无特殊画像。"

    async def increment_message_count(self, group_id: str, user_id: str):
        """原子自增 new_message_count，不存在则插入"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO user_personas (group_id, user_id, new_message_count)
                   VALUES ($1, $2, 1)
                   ON CONFLICT (group_id, user_id)
                   DO UPDATE SET new_message_count = user_personas.new_message_count + 1""",
                group_id, user_id,
            )

    async def try_acquire_persona_lock(self, group_id: str, user_id: str):
        """CAS 原子抢锁，返回行 ID 表示抢锁成功，None 表示未达阈值或已有 worker"""
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(
                """UPDATE user_personas
                   SET is_updating = TRUE, lock_acquired_at = NOW()
                   WHERE group_id = $1 AND user_id = $2
                     AND new_message_count >= 100
                     AND (is_updating = FALSE OR lock_acquired_at < NOW() - INTERVAL '5 minutes')
                   RETURNING id""",
                group_id, user_id,
            )

    async def update_persona(self, group_id: str, user_id: str, persona_text: str):
        """更新画像并释放锁"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """UPDATE user_personas
                   SET persona_text = $3, new_message_count = 0,
                       is_updating = FALSE, lock_acquired_at = NULL,
                       last_update_time = NOW()
                   WHERE group_id = $1 AND user_id = $2""",
                group_id, user_id, persona_text,
            )

    async def release_persona_lock(self, group_id: str, user_id: str):
        """异常时释放锁（不更新画像）"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """UPDATE user_personas
                   SET is_updating = FALSE, lock_acquired_at = NULL
                   WHERE group_id = $1 AND user_id = $2""",
                group_id, user_id,
            )

    async def get_recent_user_messages(self, group_id: str, user_id: str, limit: int = 50):
        """获取用户最近 N 条消息（用于人设总结）"""
        async with self.pool.acquire() as conn:
            return await conn.fetch(
                """SELECT content, timestamp FROM chat_memory
                   WHERE group_id = $1 AND user_id = $2
                   ORDER BY timestamp DESC LIMIT $3""",
                group_id, user_id, limit,
            )

    async def close(self):
        # 关闭前刷盘残余缓冲
        if self._flush_task:
            self._flush_task.cancel()
        await self._flush_buffer()
        if self.pool:
            await self.pool.close()
            logger.info("数据库连接池已关闭")
