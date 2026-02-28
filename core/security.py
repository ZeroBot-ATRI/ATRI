import re
import logging

logger = logging.getLogger("atri.security")

# 沙盒代码危险模式
SANDBOX_BLACKLIST = re.compile(
    r"os\.environ|subprocess|__import__|eval\s*\(|exec\s*\(|"
    r"open\s*\(.*/etc/|socket\.|requests\.|urllib\.",
    re.IGNORECASE,
)

# 搜索查询最大长度
MAX_SEARCH_QUERY_LEN = 200

# CQ 码匹配
CQ_CODE_PATTERN = re.compile(r"\[CQ:[^\]]+\]")

# 敏感信息模式
SENSITIVE_PATTERNS = re.compile(
    r"(sk-[a-zA-Z0-9]{20,}|key-[a-zA-Z0-9]{20,}|"
    r"system_instructions|<system>|postgresql://|"
    r"password\s*[:=]\s*\S+)",
    re.IGNORECASE,
)


def audit_sandbox_code(code: str) -> str | None:
    """审计沙盒代码，返回拦截原因或 None（通过）"""
    match = SANDBOX_BLACKLIST.search(code)
    if match:
        return f"代码包含危险调用: {match.group()}"
    return None


def audit_search_query(query: str) -> str | None:
    """审计搜索查询，返回拦截原因或 None"""
    if len(query) > MAX_SEARCH_QUERY_LEN:
        return f"查询过长（{len(query)}字符，上限{MAX_SEARCH_QUERY_LEN}）"
    return None


def audit_vision_url(url: str) -> str | None:
    """SSRF 校验：拦截内网地址"""
    import ipaddress
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        if hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
            return "禁止访问本地地址"
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback:
                return "禁止访问内网地址"
        except ValueError:
            pass
    except Exception:
        return "URL 解析失败"
    return None


def sanitize_output(text: str) -> str:
    """净化 LLM 输出：剥离 CQ 码、过滤敏感信息"""
    text = CQ_CODE_PATTERN.sub("[非法指令已拦截]", text)
    text = SENSITIVE_PATTERNS.sub("[敏感信息已过滤]", text)
    return text.strip()
