import asyncio
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import config
from pipeline.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

orchestrator = Orchestrator(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Wipe stale HLS segments from previous runs so the browser never sees
    # an old playlist while new models are loading.
    hls_path = Path(config.hls_dir)
    if hls_path.exists():
        shutil.rmtree(hls_path)
    for d in (config.hls_dir, config.clips_dir, config.audio_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    asyncio.create_task(orchestrator.run())
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/status")
async def status():
    return JSONResponse(
        {
            "ready": orchestrator.is_ready,
            "round": orchestrator._round,
            "scene": config.scene_prompt,
            "music": config.music_prompt,
        }
    )


@app.get("/hls/stream.m3u8")
async def playlist():
    path = Path(config.hls_dir) / "stream.m3u8"
    if not path.exists():
        return Response(status_code=503, content="Stream not ready yet")
    return Response(
        content=path.read_bytes(),
        media_type="application/vnd.apple.mpegurl",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/debug")
async def debug():
    hls_dir = Path(config.hls_dir)
    segments = sorted(hls_dir.glob("*.ts"))
    playlist_path = hls_dir / "stream.m3u8"
    return JSONResponse({
        "ready": orchestrator.is_ready,
        "round": orchestrator._round,
        "last_error": orchestrator.last_error,
        "playlist_exists": playlist_path.exists(),
        "playlist": playlist_path.read_text() if playlist_path.exists() else None,
        "segments": [
            {"name": s.name, "bytes": s.stat().st_size}
            for s in segments
        ],
    })


@app.get("/hls/{segment}")
async def hls_segment(segment: str):
    if not segment.endswith(".ts"):
        return Response(status_code=404)
    path = Path(config.hls_dir) / segment
    if not path.exists():
        return Response(status_code=404)
    return FileResponse(str(path), media_type="video/mp2t")
