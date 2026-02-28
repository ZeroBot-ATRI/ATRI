import logging
from ultralytics import YOLO

logger = logging.getLogger("atri.vision.detector")


class ObjectDetector:
    """YOLOv8n 物体检测与计数"""

    def __init__(self):
        self.model: YOLO | None = None

    def load(self):
        logger.info("正在加载 YOLOv8n 模型...")
        self.model = YOLO("yolov8n.pt")
        logger.info("YOLOv8n 加载完成")

    def detect(self, image_path: str) -> str:
        """检测图片中的物体，返回种类与数量的描述"""
        results = self.model(image_path, verbose=False)
        if not results:
            return "未检测到物体"

        counts = {}
        for r in results:
            for box in r.boxes:
                cls_name = r.names[int(box.cls)]
                counts[cls_name] = counts.get(cls_name, 0) + 1

        if not counts:
            return "未检测到物体"

        parts = [f"{name}: {count}个" for name, count in counts.items()]
        return "检测到: " + ", ".join(parts)
