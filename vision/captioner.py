import logging
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger("atri.vision.captioner")


class ImageCaptioner:
    """BLIP 图像场景描述"""

    def __init__(self):
        self.processor = None
        self.model = None

    def load(self):
        logger.info("正在加载 BLIP 模型...")
        model_name = "Salesforce/blip-image-captioning-base"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).cuda()
        logger.info("BLIP 加载完成")

    def describe(self, image_path: str) -> str:
        """生成图片的场景描述"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(image, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, max_new_tokens=50)
        caption = self.processor.decode(output[0], skip_special_tokens=True)
        return caption
