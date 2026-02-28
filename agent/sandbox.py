import logging
import asyncio
import uuid
import os
import base64
import tempfile
from functools import partial

import docker

logger = logging.getLogger("atri.agent.sandbox")

# 全局沙盒计数器
_running_sandboxes = 0
_sandbox_lock = asyncio.Lock()
MAX_CONCURRENT_SANDBOXES = 3


async def run_sandbox(code: str) -> dict:
    """
    代码沙盒工具：在 Docker 容器中安全执行 Python 代码
    返回 {"stdout": str, "images": [base64字符串列表]}
    """
    global _running_sandboxes

    async with _sandbox_lock:
        if _running_sandboxes >= MAX_CONCURRENT_SANDBOXES:
            return {"stdout": "沙盒繁忙，请稍后再试。", "images": []}
        _running_sandboxes += 1

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, partial(_run_in_docker, code)
        )
        return result
    finally:
        async with _sandbox_lock:
            _running_sandboxes -= 1


def _run_in_docker(code: str) -> dict:
    """同步执行：创建 Docker 容器运行代码"""
    client = docker.from_env()
    run_id = uuid.uuid4().hex[:12]
    host_workspace = os.path.join(tempfile.gettempdir(), f"atri_sandbox_{run_id}")
    os.makedirs(host_workspace, exist_ok=True)

    # Windows 路径转 Docker 兼容格式: C:\Users\... -> /c/Users/...
    docker_workspace = host_workspace
    if os.name == "nt":
        docker_workspace = "/" + host_workspace.replace("\\", "/").replace(":", "", 1)
        docker_workspace = docker_workspace[0] + docker_workspace[1].lower() + docker_workspace[2:]

    # 写入代码文件
    code_path = os.path.join(host_workspace, "main.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)

    stdout_text = ""
    images = []

    container = None
    try:
        container = client.containers.create(
            "python:3.9-slim",
            ["python", "/workspace/main.py"],
            volumes={docker_workspace: {"bind": "/workspace", "mode": "rw"}},
            working_dir="/workspace",
            network_mode="none",
            mem_limit="512m",
            pids_limit=50,
            read_only=False,
            tmpfs={"/tmp": "size=64m"},
        )
        container.start()
        result = container.wait(timeout=10)
        stdout_text = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace").strip()
        if result["StatusCode"] != 0:
            stdout_text = f"代码执行出错:\n{stdout_text}"
        if len(stdout_text) > 2000:
            stdout_text = stdout_text[:2000] + "\n...(输出已截断)"
    except Exception as e:
        stdout_text = f"沙盒执行异常: {e}"
    finally:
        if container:
            try:
                container.remove(force=True)
            except Exception:
                pass

    # 扫描输出图片并转 Base64
    images = _collect_images(host_workspace)

    # 清理临时目录
    _cleanup_workspace(host_workspace)

    return {"stdout": stdout_text, "images": images}


def _collect_images(workspace: str) -> list[str]:
    """扫描工作目录中的图片，转为 Base64（最多3张，单张<10MB）"""
    allowed = (".png", ".jpg")
    images = []
    for fname in os.listdir(workspace):
        if not fname.lower().endswith(allowed):
            continue
        fpath = os.path.join(workspace, fname)
        if os.path.getsize(fpath) > 10 * 1024 * 1024:
            continue
        with open(fpath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        images.append(b64)
        if len(images) >= 3:
            break
    return images


def _cleanup_workspace(workspace: str):
    """清理临时工作目录"""
    import shutil
    try:
        shutil.rmtree(workspace)
    except Exception as e:
        logger.warning(f"清理临时目录失败: {e}")
