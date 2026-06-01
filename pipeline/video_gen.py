import logging
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# SVD expects landscape 1024×576
SVD_WIDTH, SVD_HEIGHT = 1024, 576


class VideoGenerator:
    """
    Animates a scene image into a short looping clip.
    Uses Stable Video Diffusion XT when CUDA is available; falls back to a
    static-image clip otherwise.

    SVD takes no text prompt — motion is inferred entirely from the image,
    which is ideal for subtle lofi ambient loops.
    """

    def __init__(self, model_id: str, num_frames: int = 25, fps: int = 7, motion_bucket_id: int = 80):
        self.num_frames = num_frames
        self.fps = fps
        self.motion_bucket_id = motion_bucket_id
        self.pipe = None

        if torch.cuda.is_available():
            self._load_svd(model_id)
        else:
            logger.warning("CUDA not available — using static-image video fallback")

    def _load_svd(self, model_id: str):
        try:
            from diffusers import StableVideoDiffusionPipeline

            logger.info("Loading SVD model: %s", model_id)
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                variant="fp16",
            )
            # Sequential offload keeps each sub-module on CPU until needed,
            # minimising peak VRAM. Slower than model offload but fits 12 GB.
            self.pipe.enable_sequential_cpu_offload()
            logger.info("SVD ready")
        except Exception as exc:
            logger.error("Failed to load SVD: %s", exc, exc_info=True)
            logger.warning("SVD unavailable — falling back to static-image clip (no animation)")
            self.pipe = None

    # ------------------------------------------------------------------

    def generate_clip(self, scene_image: Image.Image, output_path: Path) -> Path:
        """Generate a looping video clip from `scene_image`."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.pipe is not None:
            return self._generate_svd(scene_image, output_path)
        return self._generate_static(scene_image, output_path)

    # ------------------------------------------------------------------

    def _generate_svd(self, image: Image.Image, output_path: Path) -> Path:
        from diffusers.utils import export_to_video

        image = image.resize((SVD_WIDTH, SVD_HEIGHT), Image.LANCZOS).convert("RGB")
        result = self.pipe(
            image,
            num_frames=self.num_frames,
            motion_bucket_id=self.motion_bucket_id,
            noise_aug_strength=0.02,
            decode_chunk_size=4,
        )
        export_to_video(result.frames[0], str(output_path), fps=self.fps)
        logger.info("SVD clip saved: %s", output_path)
        return output_path

    def _generate_static(self, image: Image.Image, output_path: Path) -> Path:
        tmp_img = output_path.with_suffix(".png")
        image.resize((SVD_WIDTH, SVD_HEIGHT), Image.LANCZOS).save(str(tmp_img))

        duration = self.num_frames / self.fps
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(tmp_img),
                "-t", str(duration),
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                str(output_path),
            ],
            check=True, capture_output=True,
        )
        tmp_img.unlink(missing_ok=True)
        logger.info("Static clip saved: %s", output_path)
        return output_path

    # ------------------------------------------------------------------

    @staticmethod
    def make_placeholder_image() -> Image.Image:
        """Warm dark gradient — replace with Flux/SDXL later for a real scene."""
        w, h = SVD_WIDTH, SVD_HEIGHT
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        r = np.clip(25 + 30 * xx + 10 * (1 - yy), 0, 255).astype(np.uint8)
        g = np.clip(15 + 20 * (1 - yy), 0, 255).astype(np.uint8)
        b = np.clip(35 + 25 * (1 - yy) + 10 * xx, 0, 255).astype(np.uint8)
        return Image.fromarray(np.stack([r, g, b], axis=2))
