import logging
import asyncio
import os
from functools import partial

# os.environ["HF_HUB_OFFLINE"] = "1"  # 阻断 HuggingFace 联网请求，强制使用本地缓存模型。第一次使用先注释掉这些。
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("atri.embedding")


class EmbeddingModel:
    """Qwen3-Embedding 本地模型封装"""

    def __init__(self, config: dict):
        self.model_name = config.get("model_name", "Qwen/Qwen3-Embedding-0.6B")
        self.device = config.get("device", "cuda")
        self.model: SentenceTransformer | None = None

    def load(self):
        logger.info(f"正在加载 Embedding 模型: {self.model_name} -> {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)#本地加载模型, local_files_only=True
        logger.info("Embedding 模型加载完成")

    def _encode_sync(self, text: str) -> list[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    async def get_embedding(self, text: str) -> str:
        """异步获取文本的 1024 维向量，返回 pgvector 兼容的字符串格式"""
        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(None, partial(self._encode_sync, text))
        return str(vec)
