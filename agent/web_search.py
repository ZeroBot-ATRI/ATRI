import asyncio
import logging
from functools import partial

from ddgs import DDGS

logger = logging.getLogger("atri.agent.web_search")

# 部分版本 .text() 返回生成器，统一转为 list
SEARCH_TIMEOUT = 15
MAX_RESULTS = 5


def _search_sync(q: str) -> list:
    """同步执行搜索，返回 list[dict]。"""
    try:
        ddgs = DDGS(timeout=SEARCH_TIMEOUT)
    except TypeError:
        ddgs = DDGS()
    raw = ddgs.text(q, max_results=MAX_RESULTS)
    return list(raw) if raw is not None else []


async def run_search(query: str) -> str:
    """联网搜索工具：封装 DuckDuckGo Search，返回前5条摘要"""
    try:
        if hasattr(asyncio, "to_thread"):
            results = await asyncio.to_thread(_search_sync, query)
        else:
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, partial(_search_sync, query))

        if not results:
            return "未搜索到相关结果。"

        parts = []
        for i, r in enumerate(results, 1):
            title = (r.get("title") or "").strip()
            body = (r.get("body") or "")[:150]
            href = (r.get("href") or "").strip()
            parts.append(f"{i}. {title}\n   {body}\n   链接: {href}")

        return "\n".join(parts)

    except Exception as e:
        logger.exception("搜索失败")
        return f"搜索出错: {e}"
