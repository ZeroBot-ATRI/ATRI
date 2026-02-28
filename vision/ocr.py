import logging
from paddleocr import PaddleOCR

logger = logging.getLogger("atri.vision.ocr")


class OCREngine:
    """PaddleOCR 中英文字符提取"""

    def __init__(self):
        self.model: PaddleOCR | None = None

    def load(self):
        logger.info("正在加载 PaddleOCR 模型...")
        self.model = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        logger.info("PaddleOCR 加载完成")

    def extract_text(self, image_path: str) -> str:
        """从图片中提取文字，返回拼接后的字符串"""
        result = self.model.ocr(image_path, cls=True)
        if not result or not result[0]:
            return ""
        lines = []
        for line in result[0]:
            text = line[1][0]
            lines.append(text)
        return "\n".join(lines)
