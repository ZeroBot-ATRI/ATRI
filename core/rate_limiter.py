import time
import logging
from collections import defaultdict, deque

logger = logging.getLogger("atri.rate_limiter")


class RateLimiter:
    """滑动窗口限流 + 全局熔断"""

    def __init__(self):
        # per-user 滑动窗口: {user_id: deque of timestamps}
        self._llm_calls = defaultdict(deque)      # 每用户每分钟 6 次
        self._sandbox_calls = defaultdict(deque)   # 每用户每5分钟 2 次
        self._search_calls = defaultdict(deque)    # 每用户每分钟 3 次

        # 全局熔断
        self._api_failures = deque()  # 1分钟内失败时间戳
        self._circuit_open_until = 0.0

    def _check_window(self, window: deque, user_id: str,
                      max_calls: int, window_secs: float) -> bool:
        """通用滑动窗口检查，返回 True 表示允许"""
        now = time.time()
        q = window[user_id]
        while q and q[0] < now - window_secs:
            q.popleft()
        if len(q) >= max_calls:
            return False
        q.append(now)
        return True

    def check_llm(self, user_id: str) -> bool:
        """LLM 调用限流：每用户每分钟 6 次"""
        return self._check_window(self._llm_calls, user_id, 6, 60)

    def check_sandbox(self, user_id: str) -> bool:
        """沙盒限流：每用户每5分钟 2 次"""
        return self._check_window(self._sandbox_calls, user_id, 2, 300)

    def check_search(self, user_id: str) -> bool:
        """搜索限流：每用户每分钟 3 次"""
        return self._check_window(self._search_calls, user_id, 3, 60)

    def is_circuit_open(self) -> bool:
        """全局熔断器是否打开"""
        return time.time() < self._circuit_open_until

    def record_api_failure(self):
        """记录一次 API 失败，超过阈值触发熔断"""
        now = time.time()
        self._api_failures.append(now)
        while self._api_failures and self._api_failures[0] < now - 60:
            self._api_failures.popleft()
        if len(self._api_failures) >= 10:
            self._circuit_open_until = now + 60
            self._api_failures.clear()
            logger.warning("全局熔断器已触发，60秒冷却期")

    def record_api_success(self):
        """API 成功时清理失败计数"""
        pass
