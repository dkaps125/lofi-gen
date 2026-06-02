import logging
from pathlib import Path

import torch
import torchaudio

logger = logging.getLogger(__name__)

# MusicGen-small uses EnCodec at 32 kHz with a 50 Hz frame rate (per the MusicGen paper).
# max_new_tokens maps 1:1 to EnCodec frames, so 1 second = 50 tokens.
_TOKENS_PER_SECOND = 50


class AudioGenerator:
    def __init__(self, model_name: str = "facebook/musicgen-small"):
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        logger.info("Loading MusicGen: %s", model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Keep on CPU at rest; move to GPU only during generate() to share VRAM with Wan
        self.model.to("cpu")

        self.sample_rate: int = self.model.config.audio_encoder.sampling_rate
        logger.info("MusicGen ready — sample_rate=%d", self.sample_rate)

    def generate(self, prompt: str, duration: int, output_path: Path) -> Path:
        max_tokens = duration * _TOKENS_PER_SECOND

        inputs = self.processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )

        self.model.to(self.device)
        try:
            inputs = {
                k: v.to(self.device, dtype=torch.float16) if v.is_floating_point() else v.to(self.device)
                for k, v in inputs.items()
            }
            with torch.inference_mode():
                audio_values = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.95,
                    top_k=250,
                    max_new_tokens=max_tokens,
                    min_new_tokens=max_tokens,
                )
        finally:
            self.model.to("cpu")
            torch.cuda.empty_cache()

        wav = audio_values[0].cpu().to(torch.float32)
        actual_dur = wav.shape[-1] / self.sample_rate
        logger.info("Audio generated: %.2fs (target %ds)", actual_dur, duration)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), wav, sample_rate=self.sample_rate, backend="ffmpeg")
        return output_path
