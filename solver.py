"""Plate-solving fallback using astrometry.net's solve-field.

Index files are downloaded on first use (only those matching the image's field of
view) into ASTROMETRY_INDEX_DIR (default ./astrometry-index).
"""
import os, shutil, subprocess
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits
from astropy.coordinates import Angle

INDEX_DIR = Path(os.environ.get("ASTROMETRY_INDEX_DIR", Path(__file__).resolve().parent / "astrometry-index"))
INDEX_URL = "http://data.astrometry.net/4100/index-{n}.fits"
# 4100-series (Tycho-2) index scale ranges, in arcminutes (skymark diameter)
RANGES = {4107: (22, 30), 4108: (30, 42), 4109: (42, 60), 4110: (60, 85), 4111: (85, 120),
          4112: (120, 170), 4113: (170, 240), 4114: (240, 340), 4115: (340, 480),
          4116: (480, 680), 4117: (680, 1000), 4118: (1000, 1400), 4119: (1400, 2000)}


def _pixel_scale(hdr):
    """Estimated arcsec/pixel from focal length and pixel size, if present."""
    f, px = hdr.get("FOCALLEN"), hdr.get("XPIXSZ")
    if f and px:
        return 206.265 * float(px) * int(hdr.get("XBINNING", 1) or 1) / float(f)
    return None


def _hint(hdr):
    try:
        if hdr.get("RA") is not None and hdr.get("DEC") is not None:
            return float(hdr["RA"]), float(hdr["DEC"])
        if hdr.get("OBJCTRA") and hdr.get("OBJCTDEC"):
            return (Angle(hdr["OBJCTRA"], unit="hourangle").deg, Angle(hdr["OBJCTDEC"], unit="deg").deg)
    except Exception:  # noqa: BLE001
        pass
    return None


def ensure_indexes(fov_w_arcmin, fov_h_arcmin, log=print):
    lo, hi = 0.1 * min(fov_w_arcmin, fov_h_arcmin), max(fov_w_arcmin, fov_h_arcmin)
    wanted = [n for n, (a, b) in RANGES.items() if b >= lo and a <= hi] or [4110, 4111, 4112]
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    for n in wanted:
        dest = INDEX_DIR / f"index-{n}.fits"
        if dest.exists():
            continue
        log(f"  downloading star index {n} (one-time)…")
        tmp = dest.with_suffix(".part")
        with requests.get(INDEX_URL.format(n=n), stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        tmp.rename(dest)
    return wanted


def solve(hdr, data, workdir, log=print):
    """Return a FITS header with a WCS for (hdr, data), or None if solving failed."""
    if not shutil.which("solve-field"):
        raise RuntimeError("solve-field (astrometry.net) is not installed; plate-solve the FITS first "
                           "(e.g. in Siril) or run inside the Docker image.")
    W, H = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])
    scale = _pixel_scale(hdr)
    if scale:
        ensure_indexes(W * scale / 60, H * scale / 60, log)
    else:
        ensure_indexes(60, 40, log)  # unknown optics: guess a ~1° field

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    a = np.nan_to_num(np.asarray(data, dtype=np.float32))
    lum = a.mean(axis=0) if a.ndim == 3 else a  # keep FITS row order so the WCS matches FITS pixels
    src = workdir / "solve-input.fits"
    fits.PrimaryHDU(lum.astype(np.float32)).writeto(src, overwrite=True)
    cfg = workdir / "astrometry.cfg"
    cfg.write_text(f"inparallel\ncpulimit 300\nadd_path {INDEX_DIR}\nautoindex\n")

    cmd = ["solve-field", str(src), "--config", str(cfg), "--dir", str(workdir), "--overwrite",
           "--no-plots", "--no-verify", "--downsample", "2", "--cpulimit", "300",
           "--new-fits", "none", "--index-xyls", "none", "--rdls", "none", "--match", "none",
           "--corr", "none", "--solved", "none", "--temp-axy", "--crpix-center", "--tweak-order", "3"]
    if scale:
        cmd += ["--scale-units", "arcsecperpix", "--scale-low", f"{scale * 0.8:.3f}",
                "--scale-high", f"{scale * 1.2:.3f}"]
    hint = _hint(hdr)
    if hint:
        cmd += ["--ra", f"{hint[0]:.5f}", "--dec", f"{hint[1]:.5f}", "--radius", "10"]
    log("  running solve-field…")
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    wcs_file = workdir / "solve-input.wcs"
    if p.returncode != 0 or not wcs_file.exists():
        log("  solve-field output:\n" + (p.stdout[-1500:] + p.stderr[-1500:]))
        return None
    wh = fits.getheader(wcs_file)
    log(f"  solved: RA {wh['CRVAL1']:.4f} Dec {wh['CRVAL2']:.4f}")
    return wh
