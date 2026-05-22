"""
Jarvis Web UI — FastAPI application.

Exposes a browser-based interface for the Jarvis viral video pipeline.
Users upload a video, choose options, then watch real-time progress
while the pipeline runs in a background thread.
"""

import os
import uuid
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from jarvis.opus_mode.pipeline import run_pipeline, PipelineResult

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = Path("data/uploads")

app = FastAPI(title="Jarvis — Viral Video Editor")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Mount static files only if the directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Thread pool for background jobs (limit concurrency to avoid OOM)
_executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

@dataclass
class ClipInfo:
    rank: int
    title: str
    score: float
    grade: str
    duration: float
    filename: str
    verdict: str = ""
    best_platform: str = ""
    predicted_views: str = ""


@dataclass
class JobStatus:
    id: str
    status: str = "pending"          # pending | running | done | error
    progress: int = 0                # 0-100
    step: str = "Queued..."
    clips: list[ClipInfo] = field(default_factory=list)
    error: str = ""
    input_filename: str = ""


# In-memory job registry
jobs: dict[str, JobStatus] = {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index(request: Request):
    """Serve the main upload / dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process")
async def process_video(
    request: Request,
    video: UploadFile = File(...),
    n_clips: int = Form(10),
    export_n: int = Form(5),
    style: str = Form("bold"),
    platform: str = Form("tiktok"),
):
    """
    Accept a video upload and queue it for pipeline processing.

    Returns immediately with a job_id that the client polls via /status/{job_id}.
    """
    job_id = uuid.uuid4().hex

    # Resolve target dimensions from platform choice
    platform_dims = {
        "tiktok":          (1080, 1920),
        "instagram":       (1080, 1920),
        "youtube_shorts":  (1080, 1920),
        "landscape":       (1920, 1080),
        "square":          (1080, 1080),
    }
    target_width, target_height = platform_dims.get(platform, (1080, 1920))

    # Save the uploaded file
    upload_dir = UPLOADS_DIR / job_id
    clips_dir = upload_dir / "clips"
    upload_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    video_path = upload_dir / "original.mp4"
    with open(video_path, "wb") as fh:
        shutil.copyfileobj(video.file, fh)

    # Register the job
    job = JobStatus(id=job_id, input_filename=video.filename or "video.mp4")
    jobs[job_id] = job

    # Submit background work
    _executor.submit(
        _run_job,
        job_id=job_id,
        video_path=str(video_path),
        output_dir=str(clips_dir),
        n_clips=n_clips,
        export_n=export_n,
        style=style,
        target_width=target_width,
        target_height=target_height,
    )

    return JSONResponse({"job_id": job_id})


@app.get("/status/{job_id}")
async def job_status(job_id: str):
    """
    Return current job progress.

    Response schema:
        {
            "status": "running" | "done" | "error",
            "progress": 0-100,
            "step": "Transcribing...",
            "clips": [...],
            "error": ""
        }
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JSONResponse({
        "status":   job.status,
        "progress": job.progress,
        "step":     job.step,
        "error":    job.error,
        "clips": [
            {
                "rank":             c.rank,
                "title":            c.title,
                "score":            c.score,
                "grade":            c.grade,
                "duration":         c.duration,
                "filename":         c.filename,
                "verdict":          c.verdict,
                "best_platform":    c.best_platform,
                "predicted_views":  c.predicted_views,
            }
            for c in job.clips
        ],
    })


@app.get("/download/{job_id}/{filename}")
async def download_clip(job_id: str, filename: str):
    """Serve a processed clip file for download."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Sanitise filename — no path traversal
    safe_name = Path(filename).name
    clip_path = UPLOADS_DIR / job_id / "clips" / safe_name

    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip not found")

    return FileResponse(
        path=str(clip_path),
        media_type="video/mp4",
        filename=safe_name,
    )


@app.get("/jobs")
async def list_jobs():
    """Return a summary of all jobs (newest first)."""
    summary = []
    for job in reversed(list(jobs.values())):
        summary.append({
            "id":             job.id,
            "status":         job.status,
            "progress":       job.progress,
            "step":           job.step,
            "clip_count":     len(job.clips),
            "input_filename": job.input_filename,
        })
    return JSONResponse(summary)


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------

def _run_job(
    job_id: str,
    video_path: str,
    output_dir: str,
    n_clips: int,
    export_n: int,
    style: str,
    target_width: int,
    target_height: int,
) -> None:
    """
    Execute the Jarvis pipeline in a background thread.
    Updates the JobStatus object as work progresses.
    """
    job = jobs[job_id]
    job.status = "running"
    job.step = "Starting..."
    job.progress = 0

    def on_progress(step: str, pct: float) -> None:
        job.step = step
        job.progress = int(min(pct, 100))

    try:
        result: PipelineResult = run_pipeline(
            video_path=video_path,
            output_dir=output_dir,
            n_clips=n_clips,
            top_n_export=export_n,
            style=style,
            target_width=target_width,
            target_height=target_height,
            on_progress=on_progress,
        )

        # Convert ProcessedClip objects to serialisable ClipInfo
        for processed in result.clips:
            clip_filename = Path(processed.output_path).name
            job.clips.append(ClipInfo(
                rank=processed.rank,
                title=processed.moment.title,
                score=processed.report.overall_score,
                grade=processed.report.grade,
                duration=processed.duration,
                filename=clip_filename,
                verdict=processed.report.verdict,
                best_platform=processed.report.best_platform,
                predicted_views=processed.report.predicted_views_range,
            ))

        job.status = "done"
        job.progress = 100
        job.step = f"Done — {len(job.clips)} clips ready"

    except Exception as exc:  # noqa: BLE001
        job.status = "error"
        job.error = str(exc)
        job.step = "Error"
