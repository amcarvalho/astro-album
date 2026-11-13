#!/usr/bin/env python3
"""Annotated astrophoto album builder for DWARF 3 (or any FITS astrophoto).

Usage:
  python album.py add <folder-or-fits> [--png FILE] [--main "M 31"] [--title "..."] [--maglim 11.5]
  python album.py album            # rebuild the album index only
  python album.py render           # re-apply templates to all photos (after editing templates/)

Each photo gets <site>/<slug>/index.html (interactive annotated viewer); the album
lives at <site>/index.html. The site folder defaults to ./site (env SITE_DIR).
"""
import argparse, html, json, math, os, re, sys, time
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent
SITE = Path(os.environ.get("SITE_DIR", ROOT / "site"))
TEMPLATES = ROOT / "templates"
UA = {"User-Agent": "astro-album-annotator/0.2 (personal astrophotography tool)"}
SIMBAD_TAP = ["https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
              "https://simbad.harvard.edu/simbad/sim-tap/sync"]
# object pages: the Harvard mirror (the main CDS site has had long outages)
SIMBAD_WEB = "https://simbad.harvard.edu/simbad/sim-id?Ident="

GREEK = {"alf": "α", "bet": "β", "gam": "γ", "del": "δ", "eps": "ε", "zet": "ζ", "eta": "η",
         "tet": "θ", "iot": "ι", "kap": "κ", "lam": "λ", "mu.": "μ", "nu.": "ν", "ksi": "ξ",
         "omi": "ο", "pi.": "π", "rho": "ρ", "sig": "σ", "tau": "τ", "ups": "υ", "phi": "φ",
         "chi": "χ", "psi": "ψ", "ome": "ω"}
GALAXY_TYPES = {"G", "GiG", "GiC", "GiP", "BiC", "AGN", "LIN", "SyG", "Sy1", "Sy2", "SBG", "bCG",
                "LSB", "EmG", "H2G", "rG", "IG", "PaG", "G?", "QSO"}
STAR_TYPES = ("*", "V*", "PM*", "SB*", "EB*", "**", "Em*", "Be*", "HB*", "RG*", "AB*", "Pu*", "LP*",
              "Mi*", "C*", "S*", "WD*", "Y*", "TT*", "Or*", "Ae*", "BS*", "SG*", "s*", "RR*", "Ce*",
              "dS*", "Sy*", "El*", "Ro*", "a2*", "bC*", "gD*", "BY*", "RS*", "Er*", "Fl*", "Ir*",
              "No*", "Ma*", "SN*", "WR*", "OH*", "pA*", "CH*", "HS*", "Ev*", "RB*", "Ax*", "XB*",
              "LM*", "HM*", "ZZ*", "LXB", "HXB")


# --------------------------------------------------------------------------- catalogs
def tap(query):
    last = None
    for url in SIMBAD_TAP:
        try:
            r = requests.post(url, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json",
                                         "QUERY": query}, headers=UA, timeout=90)
            if r.status_code == 200 and r.text.lstrip().startswith("{"):
                j = r.json()
                cols = [c["name"] for c in j["metadata"]]
                return [dict(zip(cols, row)) for row in j["data"]]
            last = f"{url}: HTTP {r.status_code} {r.text[:300]}"
        except Exception as e:  # noqa: BLE001
            last = f"{url}: {e}"
    raise RuntimeError(f"SIMBAD query failed: {last}")


def norm_id(s):
    return re.sub(r"\s+", " ", s.strip())


def key_id(s):
    s = s.lower().replace("messier", "m").replace("name ", "")
    return re.sub(r"[\s_\-]+", "", s)


def pretty_ident(i):
    i = norm_id(i)
    if i.startswith("NAME "):
        return i[5:]
    m = re.match(r"^\* (\S+?)(\d*) (\w{3})$", i)  # Bayer / Flamsteed: "* nu. And", "* 35 And"
    if m:
        g, num, con = m.groups()
        if g in GREEK:
            sup = "".join("⁰¹²³⁴⁵⁶⁷⁸⁹"[int(d)] for d in num.lstrip("0")) if num else ""
            return f"{GREEK[g]}{sup} {con}"
        return f"{m.group(1)}{num} {con}"
    if i.startswith("V* "):
        return i[3:]
    return i


ID_RANK = [("NAME ", 0), ("M ", 1), ("NGC ", 2), ("IC ", 3), ("* ", 4), ("V* ", 5),
           ("HD ", 6), ("HIP ", 7), ("UGC ", 8), ("PGC ", 9), ("TYC ", 10)]


def rank(i):
    for p, r in ID_RANK:
        if i.startswith(p):
            return r
    return 99


def wikidata_links(ids, log=print):
    """Map SIMBAD main ids -> English Wikipedia URL via Wikidata property P3083."""
    out = {}
    ids = sorted(set(ids))
    for k in range(0, len(ids), 150):
        chunk = ids[k:k + 150]
        values = " ".join(json.dumps(x) for x in chunk)
        q = ("SELECT ?id ?a WHERE { VALUES ?id { %s } ?i wdt:P3083 ?id . "
             "?a schema:about ?i; schema:isPartOf <https://en.wikipedia.org/> }" % values)
        for attempt in range(3):
            try:
                r = requests.post("https://query.wikidata.org/sparql",
                                  data={"query": q, "format": "json"}, headers=UA, timeout=90)
                r.raise_for_status()
                for b in r.json()["results"]["bindings"]:
                    out[b["id"]["value"]] = b["a"]["value"]
                break
            except Exception as e:  # noqa: BLE001
                log(f"  wikidata retry ({e})")
                time.sleep(2 + attempt * 3)
    return out


def wiki_summary(url):
    title = url.rsplit("/", 1)[-1]
    try:
        r = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
                         headers=UA, timeout=30)
        r.raise_for_status()
        return r.json().get("extract")
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- imaging
def _mtf(m, x):
    return ((m - 1) * x) / ((2 * m - 1) * x - m)


def render_fits(data):
    """FITS pixel array -> display RGB PIL image (top row first), auto-stretched if linear."""
    a = np.nan_to_num(np.asarray(data, dtype=np.float32))
    if a.ndim == 3 and a.shape[0] in (3, 4):
        a = np.moveaxis(a[:3], 0, -1)
    elif a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    elif a.ndim == 3 and a.shape[0] == 1:
        a = np.repeat(np.moveaxis(a, 0, -1), 3, axis=2)
    else:
        raise ValueError(f"Unsupported FITS data shape {a.shape}")
    mx = float(a.max()) or 1.0
    if mx > 1.5:  # integer-range data
        a = a / (65535.0 if mx <= 65535 else mx)
    a = np.clip(a, 0, 1)
    sample = a[::4, ::4]
    if float(np.median(sample)) < 0.08:  # linear (unstretched) data -> Siril-like autostretch
        for c in range(3):
            ch = sample[..., c]
            med = float(np.median(ch))
            mad = float(np.median(np.abs(ch - med))) * 1.4826
            lo = max(0.0, med - 2.8 * mad)
            x = np.clip((a[..., c] - lo) / max(1e-6, 1 - lo), 0, 1)
            xm = max(1e-6, (med - lo) / max(1e-6, 1 - lo))
            t = 0.25
            m = xm * (t - 1) / (2 * t * xm - t - xm)
            a[..., c] = _mtf(m, x)
    a = np.flipud(a)  # FITS stores the bottom row first
    return Image.fromarray((a * 255 + 0.5).astype(np.uint8), "RGB")


def detect_flip(img_lum, pts, log=print):
    """True if FITS y must be flipped to match the display image (checks bright stars)."""
    H, W = img_lum.shape

    def score(flip):
        vals = []
        for x, y in pts:
            yy = H - 1 - y if flip else y
            xi, yi = int(round(x)), int(round(yy))
            if 21 <= xi < W - 21 and 21 <= yi < H - 21:
                vals.append(img_lum[yi - 4:yi + 5, xi - 4:xi + 5].max()
                            - np.median(img_lum[yi - 20:yi + 21, xi - 20:xi + 21]))
        return float(np.mean(vals)) if vals else 0.0

    a, b = score(False), score(True)
    log(f"  orientation check: as-is={a:.1f}  flipped={b:.1f}")
    return b > a


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "photo"


# --------------------------------------------------------------------------- core
def annotate(fit, png=None, main=None, title=None, slug=None, maglim=11.5, workdir=None, log=print):
    """Annotate one photo and write it into the site. Returns the slug."""
    fit = Path(fit)
    log(f"Reading {fit.name}")
    with fits.open(fit) as hd:
        hdu = next(h for h in hd if h.data is not None and h.data.ndim >= 2)
        hdr, data = hdu.header, hdu.data
        W, H = int(hdr["NAXIS1"]), int(hdr["NAXIS2"])

        starless = "starless" in str(hdr.get("FILTER", "")).lower()
        if png:
            img = Image.open(png).convert("RGB")
            log(f"  display image: {Path(png).name}")
        elif starless:
            raise RuntimeError("This FITS is the starless layer (FILTER = 'VIS_Starless'), so it has no stars "
                               "to show or identify. Upload the PNG/JPG that came with it as the display image.")
        else:
            log("  rendering display image from FITS")
            img = render_fits(data)

        if "CTYPE1" in hdr and "CRVAL1" in hdr:
            wcs = WCS(hdr, naxis=2)
            log("  FITS already plate-solved")
        else:
            from solver import solve  # astrometry.net fallback
            log("  FITS has no plate solution, solving with astrometry.net…")
            if png:  # solve on the display image's stars (works even when the FITS is starless)
                lum = img.convert("L")
                if lum.size != (W, H):
                    lum = lum.resize((W, H), Image.LANCZOS)
                solve_data = np.flipud(np.asarray(lum, dtype=np.float32) / 255.0)  # back to FITS row order
            else:
                solve_data = data
            wcs_hdr = solve(hdr, solve_data, Path(workdir or fit.parent), log=log)
            if wcs_hdr is None:
                raise RuntimeError("Plate solving failed: could not work out where in the sky this image is.")
            wcs = WCS(wcs_hdr, naxis=2)
    if img.size != (W, H):
        log(f"  note: image {img.size} differs from FITS {W}x{H}; scaling coordinates")
    sx, sy = img.size[0] / W, img.size[1] / H

    # field footprint for the catalog query
    xs = [0, W / 2, W - 1, W - 1, W - 1, W / 2, 0, 0]
    ys = [0, 0, 0, H / 2, H - 1, H - 1, H - 1, H / 2]
    ra, dec = wcs.pixel_to_world_values(xs, ys)
    cra, cdec = (float(v) for v in wcs.pixel_to_world_values(W / 2, H / 2))
    radius = max(math.degrees(math.acos(min(1, math.sin(math.radians(cdec)) * math.sin(math.radians(d))
                 + math.cos(math.radians(cdec)) * math.cos(math.radians(d)) * math.cos(math.radians(r - cra)))))
                 for r, d in zip(ra, dec)) * 1.02
    scale_arcsec = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600
    log(f"  center RA {cra:.4f}  Dec {cdec:.4f}  radius {radius:.2f}°  {scale_arcsec:.2f}\"/px")

    log("Querying SIMBAD for stars and deep-sky objects…")
    region = f"CONTAINS(POINT('ICRS',b.ra,b.dec),CIRCLE('ICRS',{cra},{cdec},{radius}))=1"
    cols = ("b.oid,b.main_id,b.ra,b.dec,b.otype,b.galdim_majaxis,b.galdim_minaxis,b.galdim_angle,"
            "f.V,f.G,f.B")
    stars = tap(f"SELECT {cols} FROM basic b LEFT JOIN allfluxes f ON b.oid=f.oidref "
                f"WHERE {region} AND (f.V<{maglim} OR f.G<{maglim})")
    dsos = tap(f"SELECT DISTINCT {cols} FROM basic b JOIN ident i ON i.oidref=b.oid "
               f"LEFT JOIN allfluxes f ON b.oid=f.oidref WHERE {region} AND "
               "(i.id LIKE 'M %' OR i.id LIKE 'NGC %' OR i.id LIKE 'IC %')")
    objs = {o["oid"]: o for o in stars + dsos}
    dso_oids = {o["oid"] for o in dsos}
    log(f"  {len(stars)} stars, {len(dsos)} M/NGC/IC objects nearby")

    oids = list(objs)
    px, py = wcs.world_to_pixel_values([objs[o]["ra"] for o in oids], [objs[o]["dec"] for o in oids])
    lum = np.asarray(img.convert("L"), dtype=np.float32)
    bright = [(x * sx, y * sy) for o, x, y in zip(oids, px, py)
              if (objs[o]["V"] or objs[o]["G"] or 99) < 9.5]
    # our own render is always flipped to top-down; a supplied image is checked against bright stars
    flip = (detect_flip(lum, bright, log) if bright else True) if png else True
    inframe = []
    for o, x, y in zip(oids, px, py):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        yy = (H - 1 - y) if flip else y
        if -2 <= x <= W + 1 and -2 <= yy <= H + 1:
            objs[o]["x"], objs[o]["y"] = float(x), float(yy)
            inframe.append(o)

    idents = {}
    for k in range(0, len(inframe), 300):
        chunk = ",".join(str(o) for o in inframe[k:k + 300])
        for row in tap(f"SELECT oidref,id FROM ident WHERE oidref IN ({chunk})"):
            idents.setdefault(row["oidref"], []).append(norm_id(row["id"]))
    otypes = {r["otype"]: r["description"] for r in tap("SELECT otype,description FROM otypedef")}
    log("Looking up Wikipedia articles…")
    links = wikidata_links([norm_id(objs[o]["main_id"]) for o in inframe], log)
    log(f"  {len(inframe)} objects in frame, {len(links)} with Wikipedia articles")

    # main object: given name, else FITS OBJECT keyword
    main_query = (main or str(hdr.get("OBJECT", ""))).strip()
    main_oid = None
    if main_query:
        mk_ = key_id(main_query)
        for o in inframe:
            if any(key_id(i) == mk_ for i in idents.get(o, []) + [objs[o]["main_id"]]):
                main_oid = o
                break
        if main_oid is None:  # fall back to Wikipedia title match, then partial name match
            for o in inframe:
                w = links.get(norm_id(objs[o]["main_id"]), "")
                if w and key_id(requests.utils.unquote(w.rsplit("/", 1)[-1])) == mk_:
                    main_oid = o
                    break
        if main_oid is None:
            log(f"  note: '{main_query}' not matched to a catalogued object in the frame")

    markers = []
    for o in inframe:
        ob = objs[o]
        mid = norm_id(ob["main_id"])
        ids = sorted(set(idents.get(o, [mid])), key=lambda i: (rank(i), i))
        wiki = links.get(mid)
        name = pretty_ident(ids[0]) if rank(ids[0]) < 99 else mid
        if wiki:  # Wikipedia's title is the most recognisable name
            name = requests.utils.unquote(wiki.rsplit("/", 1)[-1]).replace("_", " ")
        aliases = [pretty_ident(i) for i in sorted(ids, key=lambda i: (rank(i) == 0, rank(i), i))
                   if rank(i) < 11 and pretty_ident(i) != name][:5]
        mag = ob["V"] if ob["V"] is not None else ob["G"] if ob["G"] is not None else ob["B"]
        is_star = o not in dso_oids or ob["otype"].endswith("*") or ob["otype"] in STAR_TYPES
        m = {"id": mid, "name": name, "aliases": aliases, "x": round(ob["x"] * sx, 1),
             "y": round(ob["y"] * sy, 1),
             "type": "Galaxy" if ob["otype"] in GALAXY_TYPES else otypes.get(ob["otype"], ob["otype"]),
             "kind": "star" if is_star else "dso", "mag": round(mag, 2) if mag is not None else None,
             "wiki": wiki,
             "simbad": SIMBAD_WEB + requests.utils.quote(mid),
             "main": o == main_oid}
        if not is_star and ob["galdim_majaxis"]:
            maj = ob["galdim_majaxis"] * 60 / scale_arcsec * sx
            mnr = (ob["galdim_minaxis"] or ob["galdim_majaxis"]) * 60 / scale_arcsec * sx
            if 12 < maj < 0.6 * math.hypot(*img.size):
                pa = ob["galdim_angle"] or 0
                d = 1 / 60
                r2 = ob["ra"] + d * math.sin(math.radians(pa)) / math.cos(math.radians(ob["dec"]))
                d2 = ob["dec"] + d * math.cos(math.radians(pa))
                x2, y2 = wcs.world_to_pixel_values(r2, d2)
                y2 = (H - 1 - y2) if flip else y2
                ang = math.degrees(math.atan2(y2 - ob["y"], x2 - ob["x"]))
                m["ellipse"] = [round(maj, 1), round(mnr, 1), round(ang, 1)]
        markers.append(m)

    # drop duplicates (e.g. a double star and its primary at the same spot): keep the richer entry
    markers.sort(key=lambda m: (not m["main"], not m["wiki"], m["kind"] != "dso", len(m["name"])))
    kept = []
    for m in markers:
        if not any(abs(m["x"] - k["x"]) < 3 and abs(m["y"] - k["y"]) < 3 for k in kept):
            kept.append(m)
    markers = sorted(kept, key=lambda m: (not m["main"], m["kind"] != "dso",
                                          m["mag"] if m["mag"] is not None else 99))

    main_m = next((m for m in markers if m["main"]), None)
    title = title or main or (main_m["name"] if main_m else None) or main_query or fit.stem
    slug = slugify(slug or title)
    out = SITE / slug
    out.mkdir(parents=True, exist_ok=True)
    log("Saving images…")
    img.save(out / "image.jpg", quality=90, optimize=True, progressive=True)
    th = img.copy()
    th.thumbnail((900, 900))
    th.save(out / "thumb.jpg", quality=85, optimize=True)

    def h(k):
        v = hdr.get(k)
        return v.strip() if isinstance(v, str) else v

    exp, stack = h("EXPTIME"), h("STACKCNT")
    integ = h("LIVETIME") or (exp * stack if exp and stack else None)
    data_out = {
        "slug": slug, "title": title,
        "main": main_m and {**main_m, "summary": wiki_summary(main_m["wiki"]) if main_m.get("wiki") else None},
        "width": img.size[0], "height": img.size[1],
        "meta": {
            "date": h("DATE-OBS"), "telescope": h("TELESCOP") or "", "filter": re.sub(r"[_ ]?starless", "", h("FILTER") or "", flags=re.I),
            "exposure": exp, "frames": stack, "integration": integ, "focal": h("FOCALLEN"),
            "gain": h("GAIN"), "temp": h("DET-TEMP"), "software": h("PROGRAM"),
            "ra": cra, "dec": cdec,
            "fov": [round(W * scale_arcsec / 3600, 2), round(H * scale_arcsec / 3600, 2)],
            "scale": round(scale_arcsec, 2),
        },
        "markers": markers,
        "source": {"fits": fit.name, "image": Path(png).name if png else None},
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "data.json").write_text(json.dumps(data_out, indent=1, ensure_ascii=False))
    render_page(data_out, out)
    log(f"Done: {len(markers)} objects marked")
    build_album(log)
    return slug


def render_page(data, out):
    tpl = (TEMPLATES / "photo.html").read_text()
    (out / "index.html").write_text(tpl.replace("__TITLE__", html.escape(data["title"]))
                                    .replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False)))


def render_all(log=print):
    """Re-render every photo page from its cached data.json (no catalog queries)."""
    for d in sorted(SITE.glob("*/data.json")):
        render_page(json.loads(d.read_text()), d.parent)
        log(f"  re-rendered {d.parent.name}")
    build_album(log)


def build_album(log=print):
    SITE.mkdir(parents=True, exist_ok=True)
    photos = []
    for d in sorted(SITE.glob("*/data.json")):
        j = json.loads(d.read_text())
        photos.append({"slug": j["slug"], "title": j["title"], "meta": j["meta"],
                       "count": len(j["markers"]),
                       "mainType": j["main"]["type"] if j.get("main") else None,
                       "w": j["width"], "h": j["height"]})
    photos.sort(key=lambda p: p["meta"].get("date") or "", reverse=True)
    tpl = (TEMPLATES / "album.html").read_text()
    (SITE / "index.html").write_text(tpl.replace("/*__DATA__*/null", json.dumps(photos, ensure_ascii=False)))
    log(f"Album: {len(photos)} photos")


# --------------------------------------------------------------------------- CLI
def find_inputs(path, png_arg):
    p = Path(path).expanduser()
    if p.is_dir():
        fits_files = sorted([*p.glob("*.fit"), *p.glob("*.fits"), *p.glob("*.fts")])
        if not fits_files:
            sys.exit(f"No FITS file in {p}")
        fit = fits_files[0]
    else:
        fit = p
    if png_arg:
        return fit, Path(png_arg).expanduser()
    cands = [fit.with_suffix(".png"), fit.with_suffix(".jpg"), *sorted(fit.parent.glob("*.png")),
             *sorted(fit.parent.glob("*.jpg"))]
    return fit, next((c for c in cands if c.exists()), None)  # None -> render from FITS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="annotate a photo and add it to the album")
    a.add_argument("path", help="folder containing the .fit (+ optional .png), or the .fit file")
    a.add_argument("--png", help="display image (default: PNG next to the FITS, else rendered from FITS)")
    a.add_argument("--main", help='main object, e.g. "Andromeda Galaxy" or "M 31" (default: FITS OBJECT)')
    a.add_argument("--title", help="photo title (default: the main object's name)")
    a.add_argument("--slug", help="folder name inside the site")
    a.add_argument("--maglim", type=float, default=11.5, help="faintest star magnitude to mark (default 11.5)")
    sub.add_parser("album", help="rebuild the album index")
    sub.add_parser("render", help="re-apply the HTML templates to all photos (no catalog queries)")
    args = ap.parse_args()
    if args.cmd == "add":
        fit, png = find_inputs(args.path, args.png)
        annotate(fit, png, args.main, args.title, args.slug, args.maglim)
    elif args.cmd == "render":
        render_all()
    else:
        build_album()


if __name__ == "__main__":
    main()
