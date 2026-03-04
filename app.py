import asyncio
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("atri")


def load_config() -> dict:
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)


# [图像-暂时禁用] 本地视觉模型（OCR/BLIP/YOLO）
# def load_vision_pipeline(multimodal: bool) -> dict | None:
#     """加载本地视觉模型（多模态模式下跳过）"""
#     if multimodal:
#         logger.info("多模态 LLM 已启用，跳过本地视觉模型加载")
#         return None
#     from vision.ocr import OCREngine
#     from vision.detector import ObjectDetector
#     from vision.captioner import ImageCaptioner
#     ocr = OCREngine()
#     ocr.load()
#     detector = ObjectDetector()
#     detector.load()
#     captioner = ImageCaptioner()
#     captioner.load()
#     return {"ocr": ocr, "detector": detector, "captioner": captioner}


async def main():
    config = load_config()
    multimodal = config["llm"].get("multimodal", False)

    # 1. 数据库
    from memory.database import Database
    db = Database(config["database"])
    await db.connect()
    await db.init_tables()
    logger.info("数据库就绪")

    # 2. Embedding 模型
    from memory.embedding import EmbeddingModel
    embedding = EmbeddingModel(config["embedding"])
    embedding.load()
    logger.info("Embedding 模型就绪")

    # 3. 视觉流水线 [图像-暂时禁用]
    vision_pipeline = None  # load_vision_pipeline(multimodal)

    # 4. LLM 客户端
    from agent.llm import LLMClient
    llm = LLMClient(config["llm"])
    logger.info(f"LLM 客户端就绪: {config['llm']['model']}")

    # 5. MCP 工具服务器
    from agent.mcp_tools import create_mcp_server
    mcp_server = create_mcp_server(db, embedding, config)

    # 6. 人设管理
    from memory.persona import PersonaManager
    persona = PersonaManager(db, llm)

    # 7. 启动 Bot
    from core.bot import Bot
    bot = Bot(
        config, db, embedding, llm, mcp_server,
        persona, vision_pipeline,
    )

    logger.info("ATRI 启动中...")
    try:
        await bot.run()
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
