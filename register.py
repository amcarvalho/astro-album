"""Align a display image (PNG/JPG) to the FITS frame by matching stars.

The PNG that comes with a FITS is often cropped, resized, rotated or mirrored
relative to it. Given star positions in FITS display coordinates (from the
catalog) and the PNG, find the similarity transform FITS -> PNG.
"""
import math

import numpy as np
from PIL import Image, ImageFilter


def detect_stars(img, max_stars=80):
    """Brightest point sources in an image, as an (N, 2) array of x, y (brightest first)."""
    lum = img.convert("L")
    small = 1.0
    if max(lum.size) > 2400:  # keep it fast on big frames
        small = 2400 / max(lum.size)
        lum = lum.resize((round(lum.size[0] * small), round(lum.size[1] * small)), Image.BILINEAR)
    a = np.asarray(lum, dtype=np.float32)
    bg = np.asarray(lum.filter(ImageFilter.GaussianBlur(12)), dtype=np.float32)
    hp = a - bg
    noise = float(np.median(np.abs(hp - np.median(hp)))) * 1.4826 or 1.0
    peaks = np.asarray(Image.fromarray(a).filter(ImageFilter.MaxFilter(7)), dtype=np.float32)
    ys, xs = np.nonzero((a >= peaks) & (hp > 6 * noise))
    if len(xs) == 0:
        return np.zeros((0, 2))
    # drop nebula texture: require a compact peak (centre clearly above the ring around it)
    h, w = a.shape
    keep = (xs > 5) & (ys > 5) & (xs < w - 6) & (ys < h - 6)
    xs, ys = xs[keep], ys[keep]
    ring = np.stack([a[ys + dy, xs + dx] for dy, dx in ((-4, 0), (4, 0), (0, -4), (0, 4), (-3, -3), (3, 3), (-3, 3), (3, -3))])
    compact = (a[ys, xs] - ring.max(axis=0)) > 3 * noise
    xs, ys = xs[compact], ys[compact]
    # saturated stars give a flat plateau of equal maxima: keep one point per plateau
    taken = np.zeros_like(a, dtype=bool)
    cands = []
    for x, y in zip(xs, ys):
        if taken[y, x]:
            continue
        taken[max(0, y - 3):y + 4, max(0, x - 3):x + 4] = True
        cands.append((x, y))
    # rank by total light (stretched images clip bright peaks, so peak height ranks poorly)
    gy, gx = np.mgrid[-3:4, -3:4]
    out = []
    for x, y in cands:
        box = hp[y - 3:y + 4, x - 3:x + 4].clip(0)
        s = float(box.sum()) or 1.0
        out.append((s, (x + (box * gx).sum() / s) / small, (y + (box * gy).sum() / s) / small))
    out.sort(key=lambda r: -r[0])
    return np.array([(x, y) for _, x, y in out[:max_stars]]).reshape(-1, 2)


def _similarity(src, dst):
    """Least-squares similarity transform src -> dst (rotation, uniform scale, translation)."""
    sm, dm = src.mean(0), dst.mean(0)
    s, d = src - sm, dst - dm
    a = (s[:, 0] * d[:, 0] + s[:, 1] * d[:, 1]).sum()
    b = (s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]).sum()
    n = (s ** 2).sum() or 1
    a, b = a / n, b / n
    M = np.array([[a, -b], [b, a]])
    return M, dm - M @ sm


def apply(T, xy):
    M, t = T
    return np.asarray(xy, dtype=float) @ M.T + t


def find_transform(src, dst, tol=3.0, log=print):
    """Find the transform mapping src points onto dst points.

    src: (N, 2) catalog stars, brightest first; dst: (M, 2) detected stars, brightest first.
    Returns ((M, t), mirrored, n_matches, rms) or None. M may include a mirror (det < 0).
    """
    if len(src) < 4 or len(dst) < 4:
        return None
    best = None
    for mirrored in (False, True):
        s_pts = src * [-1, 1] if mirrored else src
        S, D = s_pts[:50], dst[:80]
        i, j = np.triu_indices(len(S), 1)
        k, l = np.nonzero(~np.eye(len(D), dtype=bool))
        vs = S[j] - S[i]
        vd = D[l] - D[k]
        ls = np.log(np.hypot(vs[:, 0], vs[:, 1]) + 1e-9)
        ld = np.log(np.hypot(vd[:, 0], vd[:, 1]) + 1e-9)
        as_ = np.arctan2(vs[:, 1], vs[:, 0])
        ad = np.arctan2(vd[:, 1], vd[:, 0])
        # only pairs long enough to give a stable scale/angle
        okS, okD = np.hypot(*vs.T) > 40, np.hypot(*vd.T) > 15
        ls, as_, ld, ad = ls[okS], as_[okS], ld[okD], ad[okD]
        if len(ls) == 0 or len(ld) == 0:
            continue
        dlog = (ld[None, :] - ls[:, None]).ravel()
        dang = ((ad[None, :] - as_[:, None] + np.pi) % (2 * np.pi) - np.pi).ravel()
        sel = np.abs(dlog) < math.log(20)
        dlog, dang = dlog[sel], dang[sel]
        H, xe, ye = np.histogram2d(dlog, dang, bins=(int(2 * math.log(20) / 0.015), 360),
                                   range=((-math.log(20), math.log(20)), (-np.pi, np.pi)))
        # smooth over neighbouring bins so a value on a bin edge doesn't split its votes
        Hs = H + np.roll(H, 1, 0) + np.roll(H, 1, 1) + np.roll(np.roll(H, 1, 0), 1, 1)
        # try the strongest few (scale, angle) peaks; keep the one with most star matches
        for flat in np.argsort(Hs.ravel())[::-1][:10]:
            bi, bj = np.unravel_index(flat, H.shape)
            scale = math.exp(xe[bi])  # centre of the 2x2 smoothed cell = shared bin edge
            ang = ye[bj]
            M = scale * np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
            # translation vote
            cand = (D[None, :, :] - (S @ M.T)[:, None, :]).reshape(-1, 2)
            cell = 4.0
            keys, counts = np.unique(np.round(cand / cell).astype(np.int64), axis=0, return_counts=True)
            t = keys[np.argmax(counts)] * cell
            T = (M, t.astype(float))
            T, n, rms = _refine(s_pts, dst, T, tol)
            if n and (best is None or n > best[2]):
                best = (T, mirrored, n, rms)
    if best is None:
        return None
    (M, t), mirrored, n, rms = best
    if mirrored:  # fold the x-mirror into the matrix so callers apply one transform
        M = M @ np.diag([-1.0, 1.0])
    return (M, t), mirrored, n, rms


def _refine(src, dst, T, tol, iters=4):
    n, rms = 0, None
    for r in (tol * 3, tol * 2, tol, tol):
        p = apply(T, src)
        d = np.hypot(p[:, None, 0] - dst[None, :, 0], p[:, None, 1] - dst[None, :, 1])
        nn = d.argmin(1)
        dist = d[np.arange(len(src)), nn]
        m = dist < r
        # one-to-one: a detected star may only match its closest catalog star
        if m.sum() < 4:
            return T, 0, None
        idx = np.nonzero(m)[0]
        _, first = np.unique(nn[idx][np.argsort(dist[idx])], return_index=True)
        idx = idx[np.argsort(dist[idx])][first]
        T = _similarity(src[idx], dst[nn[idx]])
        n = len(idx)
        rms = float(np.sqrt(np.mean(np.hypot(*(apply(T, src[idx]) - dst[nn[idx]]).T) ** 2)))
    return T, n, rms
