import logging
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    path: Path
    duration: float
    sequence: int


class HLSManager:
    """
    Combines a (looping) video clip with an audio file via FFmpeg, segments the
    result into MPEG-TS chunks, and maintains a live HLS playlist.

    Key design: each FFmpeg call receives an -output_ts_offset equal to the
    cumulative stream duration so far.  This gives every segment monotonically
    increasing PTS across the whole stream, which is required for MSE/HLS.js to
    append segments without timestamp collisions.
    """

    def __init__(self, output_dir: Path, segment_duration: int = 6, window_size: int = 12):
        self.output_dir = output_dir
        self.segment_duration = segment_duration
        self.window_size = window_size
        self._segments: list[Segment] = []
        self._global_seq: int = 0
        self._ts_offset: float = 0.0   # cumulative PTS offset in seconds
        self._lock = threading.Lock()
        output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_round(self, video_clip: Path, audio_file: Path, audio_duration: float) -> list[Segment]:
        base = self._global_seq
        seg_pattern = str(self.output_dir / f"seg{base:07d}_%03d.ts")

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-stream_loop", "-1", "-i", str(video_clip),
                "-i", str(audio_file),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-force_key_frames", f"expr:gte(t,n_forced*{self.segment_duration})",
                "-c:a", "aac", "-b:a", "128k",
                "-t", str(audio_duration),
                "-pix_fmt", "yuv420p",
                # Offset output PTS so segments are continuous across rounds.
                # Do NOT use -reset_timestamps: that resets PTS per segment,
                # causing MSE timestamp collisions and playback stalls.
                "-output_ts_offset", str(self._ts_offset),
                "-f", "segment",
                "-segment_time", str(self.segment_duration),
                "-segment_format", "mpegts",
                seg_pattern,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        new_paths = sorted(self.output_dir.glob(f"seg{base:07d}_*.ts"))
        new_segs = [
            Segment(path=p, duration=self._probe_duration(p), sequence=self._global_seq + i)
            for i, p in enumerate(new_paths)
        ]

        with self._lock:
            self._segments.extend(new_segs)
            self._global_seq += len(new_segs)
            actual_duration = sum(s.duration for s in new_segs)
            self._ts_offset += actual_duration
            self._evict_old_segments()
            self._write_playlist()

        logger.info(
            "Round added: %d new segments, ts_offset now %.1fs, playlist seq %d–%d",
            len(new_segs), self._ts_offset,
            self._segments[0].sequence, self._segments[-1].sequence,
        )
        return new_segs

    @property
    def is_ready(self) -> bool:
        return len(self._segments) >= 3

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evict_old_segments(self):
        while len(self._segments) > self.window_size:
            old = self._segments.pop(0)
            old.path.unlink(missing_ok=True)

    def _write_playlist(self):
        if not self._segments:
            return

        target_dur = max(s.duration for s in self._segments)
        first_seq = self._segments[0].sequence

        lines = [
            "#EXTM3U",
            "#EXT-X-VERSION:3",
            f"#EXT-X-TARGETDURATION:{int(target_dur) + 1}",
            f"#EXT-X-MEDIA-SEQUENCE:{first_seq}",
        ]
        for seg in self._segments:
            lines.append(f"#EXTINF:{seg.duration:.3f},")
            lines.append(seg.path.name)

        content = "\n".join(lines) + "\n"

        tmp = self.output_dir / "stream.m3u8.tmp"
        tmp.write_text(content)
        tmp.rename(self.output_dir / "stream.m3u8")

    def _probe_duration(self, path: Path) -> float:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        try:
            return float(result.stdout.strip())
        except ValueError:
            return float(self.segment_duration)
