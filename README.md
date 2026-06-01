# lofi-stream

A self-hosted lofi stream that generates video and audio continuously using local AI models, served over HLS to a browser.

> **Note:** This project was generated with AI assistance (Claude).

## How it works

1. **Scene** — SDXL generates a still image from a text prompt (cozy cafe, rainy night, etc.)
2. **Video** — Stable Video Diffusion animates the still into a short looping clip
3. **Audio** — MusicGen generates music, each segment seeded from the tail of the previous one for seamless continuation
4. **Stream** — FFmpeg muxes audio + looping video into MPEG-TS segments; FastAPI serves a live HLS playlist

## Requirements

- Python 3.11
- CUDA GPU with ~12 GB VRAM (tested on RTX 5070)
- FFmpeg
- `uv`

## Setup

```bash
uv sync
```

SVD (Stable Video Diffusion) is a gated model. Before running, accept the license at [huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) and set your token:

```bash
export HF_TOKEN=your_token_here
```

## Running

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. The stream starts generating immediately; the page will show a spinner until enough segments are buffered (~2–3 minutes on first run while models download).

## Configuration

Edit `config.py` to change prompts, segment duration, video refresh interval, or model names.
