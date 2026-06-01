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

    # How long each generated audio segment is (seconds)
    audio_segment_duration: int = 30

    # HLS target segment length (seconds)
    hls_segment_duration: int = 6

    # Sliding window of segments kept in the live playlist
    playlist_window: int = 12

    # Regenerate the video clip every N audio rounds
    video_regen_interval: int = 8

    # MusicGen model: "facebook/musicgen-small" | "facebook/musicgen-medium" | "facebook/musicgen-melody"
    musicgen_model: str = "facebook/musicgen-small"

    # SDXL: text-to-image for generating lofi scene stills
    sdxl_model: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # SVD XT: image-to-video, ~1.5B params, fits in 12 GB
    svd_model: str = "stabilityai/stable-video-diffusion-img2vid-xt"

    # 25 frames at 7 fps ≈ 3.5 s loop; motion_bucket_id controls intensity (0–255)
    svd_num_frames: int = 25
    svd_fps: int = 7
    svd_motion_bucket_id: int = 80

    hls_dir: str = "output/hls"
    clips_dir: str = "output/clips"
    audio_dir: str = "output/audio"

    host: str = "0.0.0.0"
    port: int = 8000


config = Config()
