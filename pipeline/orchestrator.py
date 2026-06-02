import asyncio
import logging
import os
import subprocess
from pathlib import Path

from config import Config
from pipeline.audio_gen import AudioGenerator
from pipeline.hls_manager import HLSManager
from pipeline.wan_gen import WanT2VGenerator

logger = logging.getLogger(__name__)

_FADE_SECONDS = 4


class Orchestrator:
    """
    Two concurrent loops share a single GPU lock:
      - Audio loop: generates a new segment every ~audio_segment_duration seconds.
      - Video loop: regenerates the background clip every video_regen_interval
        audio rounds, running between audio generations so it never blocks them.
    """

    def __init__(self, config: Config):
        self.config = config
        self.audio_gen: AudioGenerator | None = None
        self.video_gen: WanT2VGenerator | None = None
        self.hls: HLSManager | None = None
        self._current_clip: Path | None = None
        self._round: int = 0
        self._ready = asyncio.Event()
        self.last_error: str | None = None
        # Prevents MusicGen and Wan from using the GPU at the same time.
        self._gpu_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def run(self):
        try:
            await self._startup()
        except Exception as exc:
            self.last_error = f"startup: {exc}"
            logger.exception("Startup failed")
            return
        while True:
            try:
                await self._generation_round()
            except Exception as exc:
                self.last_error = f"round {self._round}: {exc}"
                logger.exception("Generation round failed — retrying in 5s")
                await asyncio.sleep(5)

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _startup(self):
        loop = asyncio.get_event_loop()
        cfg = self.config

        token = os.environ.get("HF_TOKEN")
        if token:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            logger.info("Logged in to HuggingFace")

        logger.info("=== Loading models ===")
        self.audio_gen = await loop.run_in_executor(
            None, lambda: AudioGenerator(cfg.musicgen_model)
        )
        self.video_gen = await loop.run_in_executor(
            None,
            lambda: WanT2VGenerator(
                model_id=cfg.wan_model,
                num_frames=cfg.wan_num_frames,
                height=cfg.wan_height,
                width=cfg.wan_width,
                guidance_scale=cfg.wan_guidance_scale,
                num_inference_steps=cfg.wan_num_inference_steps,
            ),
        )

        self.hls = HLSManager(
            Path(cfg.hls_dir),
            segment_duration=cfg.hls_segment_duration,
            window_size=cfg.playlist_window,
        )

        clip_path = Path(cfg.clips_dir) / "clip_000000.mp4"
        if clip_path.exists():
            logger.info("=== Reusing existing video clip: %s ===", clip_path)
            self._current_clip = clip_path
        else:
            logger.info("=== Generating initial video clip ===")
            async with self._gpu_lock:
                self._current_clip = await loop.run_in_executor(
                    None, lambda: self.video_gen.generate_clip(cfg.scene_prompt, clip_path)
                )

        # Start the background video refresh loop before filling the audio buffer
        # so the first regen fires on schedule.
        asyncio.create_task(self._video_loop())

        logger.info("=== Pre-filling audio buffer ===")
        for _ in range(3):
            await self._generation_round()
        self._ready.set()
        logger.info("=== Stream is live ===")

    async def _video_loop(self):
        """Regenerate the video clip in the background between audio rounds."""
        cfg = self.config
        loop = asyncio.get_event_loop()
        interval = cfg.video_regen_interval * cfg.audio_segment_duration
        while True:
            await asyncio.sleep(interval)
            logger.info("[video] Regenerating clip...")
            old_clip = self._current_clip
            clip_path = Path(cfg.clips_dir) / f"clip_{self._round:06d}.mp4"
            try:
                async with self._gpu_lock:
                    new_clip = await loop.run_in_executor(
                        None, lambda: self.video_gen.generate_clip(cfg.scene_prompt, clip_path)
                    )
                self._current_clip = new_clip
                if old_clip is not None:
                    old_clip.unlink(missing_ok=True)
                logger.info("[video] Clip updated: %s", clip_path)
            except Exception:
                logger.exception("[video] Regen failed — keeping current clip")

    async def _generation_round(self):
        loop = asyncio.get_event_loop()
        cfg = self.config
        r = self._round

        audio_path = Path(cfg.audio_dir) / f"audio_{r:06d}.wav"
        faded_path = Path(cfg.audio_dir) / f"audio_{r:06d}_faded.wav"

        try:
            logger.info("[round %d] Generating audio...", r)
            async with self._gpu_lock:
                await loop.run_in_executor(
                    None,
                    lambda: self.audio_gen.generate(
                        cfg.music_prompt, cfg.audio_segment_duration, audio_path
                    ),
                )

            await loop.run_in_executor(
                None,
                lambda: _apply_fades(audio_path, faded_path, _FADE_SECONDS, cfg.audio_segment_duration),
            )

            logger.info("[round %d] Muxing → HLS...", r)
            await loop.run_in_executor(
                None,
                lambda: self.hls.add_round(
                    self._current_clip, faded_path, float(cfg.audio_segment_duration)
                ),
            )
        finally:
            audio_path.unlink(missing_ok=True)
            faded_path.unlink(missing_ok=True)

        self._round += 1


def _apply_fades(audio: Path, output: Path, fade_seconds: int, segment_duration: int) -> None:
    fade_out_start = segment_duration - fade_seconds
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(audio),
            "-af", (
                f"afade=t=in:st=0:d={fade_seconds}:curve=hsin,"
                f"afade=t=out:st={fade_out_start}:d={fade_seconds}:curve=hsin"
            ),
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
