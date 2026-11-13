"""Astro album web server: serves the album and annotates uploaded FITS files.

Env:
  DATA_DIR              root for persistent data (default ./data)
  SITE_DIR              generated site (default $DATA_DIR/site)
  ASTROMETRY_INDEX_DIR  plate-solving star indexes (default $DATA_DIR/astrometry-index)
"""
import os
from pathlib import Path

DATA = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent / "data"))
os.environ.setdefault("SITE_DIR", str(DATA / "site"))
os.environ.setdefault("ASTROMETRY_INDEX_DIR", str(DATA / "astrometry-index"))

import queue, shutil, threading, time, traceback, uuid  # noqa: E402

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import album  # noqa: E402

UPLOADS = DATA / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
album.SITE.mkdir(parents=True, exist_ok=True)
album.render_all()  # re-apply the current templates, so image updates fix existing pages

app = FastAPI(title="Astro Album")
jobs: dict[str, dict] = {}
work: "queue.Queue[str]" = queue.Queue()


def worker():
    while True:
        jid = work.get()
        job = jobs[jid]
        job["status"] = "running"
        d = UPLOADS / jid

        def log(msg):
            job["log"].append(str(msg))

        try:
            slug = album.annotate(d / job["fits"], d / job["image"] if job["image"] else None,
                                  main=job["name"] or None, title=job["name"] or None,
                                  workdir=d / "solve", log=log)
            shutil.rmtree(d / "solve", ignore_errors=True)
            keep = UPLOADS / slug
            if keep.exists():
                shutil.rmtree(keep)
            d.rename(keep)  # keep the original upload so the photo can be re-processed later
            job.update(status="done", slug=slug)
        except Exception as e:  # noqa: BLE001
            log(traceback.format_exc(limit=3))
            job.update(status="error", error=str(e))
            shutil.rmtree(d, ignore_errors=True)
        finally:
            job["finished"] = time.time()
            work.task_done()


threading.Thread(target=worker, daemon=True).start()


@app.get("/api/health")
def health():
    return {"ok": True}


def _save(up: UploadFile, dest: Path):
    with open(dest, "wb") as fh:
        shutil.copyfileobj(up.file, fh, 1 << 20)


@app.post("/api/upload")
def upload(fits: UploadFile = File(...), name: str = Form(""), image: UploadFile | None = File(None)):
    fname = Path(fits.filename or "upload.fit").name
    if not fname.lower().endswith((".fit", ".fits", ".fts")):
        raise HTTPException(400, "Please upload a .fit / .fits file")
    name = name.strip()[:120]
    jid = uuid.uuid4().hex[:12]
    d = UPLOADS / jid
    d.mkdir(parents=True)
    _save(fits, d / fname)
    img_name = None
    if image is not None and image.filename:
        img_name = Path(image.filename).name
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            shutil.rmtree(d)
            raise HTTPException(400, "Display image must be PNG, JPG or TIFF")
        _save(image, d / img_name)
    jobs[jid] = {"id": jid, "status": "queued", "name": name, "fits": fname, "image": img_name,
                 "log": [f"Uploaded {fname}" + (f" + {img_name}" if img_name else "")],
                 "created": time.time(), "position": work.qsize()}
    work.put(jid)
    return {"id": jid}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "Unknown job")
    return job


# the generated album (mounted last so /api wins)
app.mount("/", StaticFiles(directory=album.SITE, html=True), name="site")
