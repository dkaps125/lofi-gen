import logging
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

logger = logging.getLogger(__name__)

_NEGATIVE_PROMPT = (
    "low quality, blurry, static, overexposed, bright tones, subtitles, watermark, "
    "still picture, worst quality, JPEG artifacts, deformed, disfigured"
)


class WanT2VGenerator:
    """
    Generates video clips from text prompts using Wan2.1-T2V-1.3B via diffusers.
    Model is kept loaded between calls; enable_model_cpu_offload() moves sub-modules
    to GPU only during inference so the ~8 GB footprint doesn't conflict with MusicGen.
    """

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 50,
    ):
        logger.info("Loading Wan T2V: %s", model_id)
        # VAE must stay in float32 — bfloat16 causes decode artifacts
        vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )
        self.pipe = WanPipeline.from_pretrained(
            model_id, vae=vae, torch_dtype=torch.bfloat16
        )
        self.pipe.enable_model_cpu_offload()

        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        logger.info(
            "Wan T2V ready — %dx%d %d frames %d steps",
            width, height, num_frames, num_inference_steps,
        )

    def generate_clip(self, prompt: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=_NEGATIVE_PROMPT,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
            )

        torch.cuda.empty_cache()
        export_to_video(result.frames[0], str(output_path), fps=16)
        logger.info("Wan clip saved: %s", output_path)
        return output_path
