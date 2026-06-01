import asyncio
import logging
import os
from pathlib import Path

from config import Config
from pipeline.audio_gen import AudioGenerator
from pipeline.hls_manager import HLSManager
from pipeline.scene_gen import SceneGenerator
from pipeline.video_gen import VideoGenerator

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Drives the generation loop:
      1. Load models once on startup.
      2. Generate an initial video clip and audio segment.
      3. Continuously produce new audio segments, refreshing the video periodically.
    """

    def __init__(self, config: Config):
        self.config = config
        self.audio_gen: AudioGenerator | None = None
        self.scene_gen: SceneGenerator | None = None
        self.video_gen: VideoGenerator | None = None
        self.hls: HLSManager | None = None
        self._current_clip: Path | None = None
        # Path to the previous round's audio file, used as the continuation seed.
        # Kept alive until the round after it's consumed, then deleted.
        self._prev_audio: Path | None = None
        self._round: int = 0
        self._ready = asyncio.Event()
        self.last_error: str | None = None

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
        self.scene_gen = await loop.run_in_executor(
            None, lambda: SceneGenerator(cfg.sdxl_model)
        )
        self.video_gen = await loop.run_in_executor(
            None,
            lambda: VideoGenerator(
                cfg.svd_model, cfg.svd_num_frames, cfg.svd_fps, cfg.svd_motion_bucket_id
            ),
        )

        self.hls = HLSManager(
            Path(cfg.hls_dir),
            segment_duration=cfg.hls_segment_duration,
            window_size=cfg.playlist_window,
        )

        logger.info("=== Generating initial scene ===")
        scene = await loop.run_in_executor(
            None, lambda: self.scene_gen.generate(cfg.scene_prompt)
        )
        clip_path = Path(cfg.clips_dir) / "clip_000000.mp4"
        self._current_clip = await loop.run_in_executor(
            None, lambda: self.video_gen.generate_clip(scene, clip_path)
        )

        # Fill the buffer with three rounds before going live (~18 segments / 108s)
        for _ in range(3):
            await self._generation_round()
        self._ready.set()
        logger.info("=== Stream is live ===")

    async def _generation_round(self):
        loop = asyncio.get_event_loop()
        cfg = self.config
        r = self._round

        audio_path = Path(cfg.audio_dir) / f"audio_{r:06d}.wav"
        prev_audio = self._prev_audio  # capture for lambda closure

        logger.info("[round %d] Generating audio (continuation=%s)...", r, prev_audio is not None)
        await loop.run_in_executor(
            None,
            lambda: self.audio_gen.generate(
                cfg.music_prompt,
                cfg.audio_segment_duration,
                audio_path,
                prompt_path=prev_audio,
            ),
        )

        # Periodically refresh the video clip
        if r > 0 and r % cfg.video_regen_interval == 0:
            logger.info("[round %d] Regenerating video clip...", r)
            scene = await loop.run_in_executor(
                None, lambda: self.scene_gen.generate(cfg.scene_prompt)
            )
            clip_path = Path(cfg.clips_dir) / f"clip_{r:06d}.mp4"
            self._current_clip = await loop.run_in_executor(
                None,
                lambda: self.video_gen.generate_clip(scene, clip_path),
            )

        logger.info("[round %d] Muxing → HLS...", r)
        await loop.run_in_executor(
            None,
            lambda: self.hls.add_round(
                self._current_clip,
                audio_path,
                float(cfg.audio_segment_duration),
            ),
        )

        # Retire the previous round's audio now that it has been consumed,
        # then promote this round's audio as the next seed.
        if self._prev_audio is not None:
            self._prev_audio.unlink(missing_ok=True)
        self._prev_audio = audio_path
        self._round += 1
