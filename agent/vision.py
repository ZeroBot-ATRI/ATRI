import logging
import asyncio
from functools import partial

logger = logging.getLogger("atri.agent.vision")


async def run_vision(vision_pipeline: dict, image_url: str, mode: str) -> str:
    """
    图片理解工具：根据 mode 路由至对应本地模型
    - ocr: PaddleOCR 文字提取
    - detect: YOLOv8n 物体检测
    - describe: BLIP 场景描述
    """
    import httpx, tempfile, os

    # 下载图片
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()
        suffix = ".png" if "png" in image_url.lower() else ".jpg"
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(resp.content)
    except Exception as e:
        return f"图片下载失败: {e}"

    loop = asyncio.get_event_loop()
    try:
        if mode == "ocr":
            result = await loop.run_in_executor(
                None, partial(vision_pipeline["ocr"].extract_text, path)
            )
            return result if result else "未识别到文字。"

        elif mode == "detect":
            result = await loop.run_in_executor(
                None, partial(vision_pipeline["detector"].detect, path)
            )
            return result

        elif mode == "describe":
            result = await loop.run_in_executor(
                None, partial(vision_pipeline["captioner"].describe, path)
            )
            return result if result else "无法生成描述。"

        else:
            return f"不支持的模式: {mode}"
    except Exception as e:
        logger.error(f"视觉分析失败: {e}")
        return f"视觉分析出错: {e}"
    finally:
        os.remove(path)
