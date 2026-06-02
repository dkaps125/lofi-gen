# lofi-stream

A self-hosted lofi stream that generates video and audio continuously using local AI models, served over HLS to a browser.

> **Note:** This project was generated with AI assistance (Claude).

## How it works

1. **Video** — Wan2.1-T2V-1.3B generates a short looping video clip from the scene prompt
2. **Audio** — MusicGen generates music, each segment seeded from the tail of the previous one for seamless continuation
3. **Stream** — FFmpeg muxes audio + looping video into MPEG-TS segments; FastAPI serves a live HLS playlist

## Requirements

- Python 3.11
- CUDA GPU with ~12 GB VRAM (tested on RTX 5070; Wan needs ~8 GB)
- FFmpeg
- `uv`

## Setup

```bash
uv sync
```

## Running

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. The stream starts generating immediately; the page will show a spinner until enough segments are buffered (~2–3 minutes on first run while models download).

## Configuration

Edit `config.py` to change prompts, segment duration, video refresh interval, or model settings.
