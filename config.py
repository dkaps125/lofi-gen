from dataclasses import dataclass


@dataclass
class Config:
    scene_prompt: str = (
        "cozy japanese cafe at night, rain streaking down window, warm amber lamp light, "
        "steam curling from coffee cup, lo-fi aesthetic, cinematic depth of field"
    )

    music_prompt: str = (
        "lo-fi hip hop, mellow jazzy piano, soft vinyl crackle, gentle rain ambience, "
        "slow relaxing beats, warm bass, study music, 75 bpm"
    )

    # How long each generated audio segment is (seconds).
    # musicgen-small was trained on 30s clips; 60–90s still sounds good,
    # quality gets more variable beyond that. Generation time scales linearly.
    audio_segment_duration: int = 90

    # HLS target segment length (seconds)
    hls_segment_duration: int = 6

    # Sliding window of segments kept in the live playlist.
    # Should comfortably hold at least 2 full audio rounds' worth of segments:
    # 2 × ceil(audio_segment_duration / hls_segment_duration).
    playlist_window: int = 30

    # Regenerate the video clip every N audio rounds
    video_regen_interval: int = 8

    # MusicGen model: "facebook/musicgen-small" | "facebook/musicgen-medium"
    musicgen_model: str = "facebook/musicgen-small"

    # Wan2.1 T2V — 1.3B model fits comfortably in 12 GB VRAM
    wan_model: str = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    wan_num_frames: int = 81        # 81 frames at 16 fps ≈ 5 s
    wan_height: int = 480
    wan_width: int = 832
    wan_guidance_scale: float = 5.0
    wan_num_inference_steps: int = 50

    hls_dir: str = "output/hls"
    clips_dir: str = "output/clips"
    audio_dir: str = "output/audio"

    host: str = "0.0.0.0"
    port: int = 8000


config = Config()
