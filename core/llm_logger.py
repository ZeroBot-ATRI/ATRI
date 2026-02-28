"""
LLM 交互日志模块
将每次 LLM 调用的输入（messages/tools）和输出（response）保存为 JSON 文件
"""

import json
import os
from datetime import datetime

LOG_DIR = "logs/llm"


def _ensure_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _make_path(tag: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return os.path.join(LOG_DIR, f"{ts}_{tag}.json")


def save_simple_chat(payload: dict, response: dict):
    """记录 simple_chat 的一次请求/响应"""
    _ensure_dir()
    record = {
        "type": "simple_chat",
        "timestamp": datetime.now().isoformat(),
        "request": payload,
        "response": response,
    }
    path = _make_path("simple")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def save_tool_chat(rounds: list):
    """
    记录 chat_with_tools 的完整多轮交互

    rounds 结构: [
        {"request": payload_dict, "response": response_dict},
        ...
    ]
    """
    _ensure_dir()
    record = {
        "type": "chat_with_tools",
        "timestamp": datetime.now().isoformat(),
        "total_rounds": len(rounds),
        "rounds": rounds,
    }
    path = _make_path("tools")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
