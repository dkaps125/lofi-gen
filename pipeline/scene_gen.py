import logging

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# SVD target resolution
_WIDTH, _HEIGHT = 1024, 576

_NEGATIVE = (
    "blurry, low quality, distorted, text, watermark, signature, nsfw, "
    "photorealistic person, crowd, ugly, deformed"
)


class SceneGenerator:
    """
    Generates a still scene image from a text prompt using SDXL.
    The image is then passed to VideoGenerator (SVD) to be animated.
    """

    def __init__(self, model_id: str):
        from diffusers import DiffusionPipeline

        logger.info("Loading SDXL: %s", model_id)
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.pipe.enable_sequential_cpu_offload()
        logger.info("SDXL ready")

    def generate(self, prompt: str) -> Image.Image:
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=_NEGATIVE,
                width=_WIDTH,
                height=_HEIGHT,
                num_inference_steps=25,
                guidance_scale=7.5,
            )
        image = result.images[0]
        logger.info("Scene image generated: %dx%d", image.width, image.height)
        return image
