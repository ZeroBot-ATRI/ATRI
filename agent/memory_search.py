import logging

logger = logging.getLogger("atri.agent.memory_search")


async def query_chat_memory(db, embedding_model, group_id: str,
                            query_text: str = None, target_user_id: str = None) -> str:
    """
    记忆检索工具：时间半衰期加权 RAG
    - query_text 不填则拉取最新记录
    - target_user_id 可选，指定用户
    """
    if query_text:
        embedding_vec = await embedding_model.get_embedding(query_text)
        rows = await db.search_memory(
            group_id, embedding_vec,
            half_life_days=1.0,
            target_user_id=target_user_id,
            limit=10,
        )
    else:
        # 无搜索词，直接拉最新
        rows = await db.get_recent_messages(group_id, limit=10)

    if not rows:
        return "未找到相关记忆。"

    results = []
    for r in rows:
        user = r["user_id"]
        name = (r.get("user_name") or "").strip()
        label = f"{name}(QQ:{user})" if name else f"QQ:{user}"
        content = r["content"][:128]
        ts = r["timestamp"].strftime("%m-%d %H:%M")
        results.append(f"[{ts}] {label}: {content}")

    return "\n".join(results)
