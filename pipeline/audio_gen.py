import logging
from pathlib import Path

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)

# MusicGen-small uses EnCodec at 32 kHz with a 50 Hz frame rate (per the MusicGen paper).
# max_new_tokens maps 1:1 to EnCodec frames, so 1 second = 50 tokens.
_TOKENS_PER_SECOND = 50

# How many seconds of the previous audio to feed back as the autoregressive seed.
_CONTINUATION_SECONDS = 5


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
        # Keep on CPU at rest; move to GPU only during generate() to share VRAM with SVD
        self.model.to("cpu")

        self.sample_rate: int = self.model.config.audio_encoder.sampling_rate
        logger.info("MusicGen ready — sample_rate=%d tokens/s=%d", self.sample_rate, _TOKENS_PER_SECOND)

    def generate(self, prompt: str, duration: int, output_path: Path,
                 prompt_path: Path | None = None) -> Path:
        """
        Generate `duration` seconds of audio.

        If `prompt_path` is given, the last _CONTINUATION_SECONDS of that file
        are fed into the MusicGen decoder as a seed so the new audio continues
        the previous segment without any crossfade.  The model output includes
        both the seed portion and the freshly generated audio; we strip the
        seed before writing so callers always get exactly `duration` seconds.
        """
        max_tokens = duration * _TOKENS_PER_SECOND

        if prompt_path is not None:
            audio_array = self._load_tail(prompt_path, _CONTINUATION_SECONDS)
            inputs = self.processor(
                audio=[audio_array],
                sampling_rate=self.sample_rate,
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )

        self.model.to(self.device)
        try:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
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

        # audio_values: [batch=1, channels, total_samples]
        # When a prompt was used the output leads with the re-encoded seed audio;
        # strip it so the saved file contains only the new material.
        wav = audio_values[0].cpu().to(torch.float32)
        if prompt_path is not None:
            prompt_samples = int(_CONTINUATION_SECONDS * self.sample_rate)
            if wav.shape[-1] > prompt_samples:
                wav = wav[:, prompt_samples:]
            else:
                logger.warning(
                    "Output (%d samples) no longer than prompt (%d samples) — "
                    "skipping trim; continuation may not have worked",
                    wav.shape[-1], prompt_samples,
                )

        actual_dur = wav.shape[-1] / self.sample_rate
        logger.info("Audio generated: %.2fs (target %ds)", actual_dur, duration)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), wav, sample_rate=self.sample_rate, backend="ffmpeg")
        return output_path

    def _load_tail(self, path: Path, seconds: int) -> np.ndarray:
        """Return the last `seconds` of audio as a mono float32 numpy array at model sample rate."""
        waveform, sr = torchaudio.load(str(path))
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        n_samples = int(seconds * self.sample_rate)
        return waveform[0, -n_samples:].numpy()
