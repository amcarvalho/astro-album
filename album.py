#!/usr/bin/env python3
"""Annotated astrophoto album builder for DWARF 3 (or any FITS astrophoto).

Usage:
  python album.py add <folder-or-fits> [--png FILE] [--main "M 31"] [--title "..."] [--maglim 11.5]
  python album.py album            # rebuild the album index only
  python album.py render           # re-apply templates to all photos (after editing templates/)
  python album.py reprocess [slug] # re-tag photos from their saved uploads (after a fix)

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

import register

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parent
SITE = Path(os.environ.get("SITE_DIR") or (Path(os.environ["DATA_DIR"]) / "site" if os.environ.get("DATA_DIR") else ROOT / "site"))
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
CLUSTER_TYPES = {"Cl*", "OpC", "GlC", "As*", "St*", "MGr", "C?*", "Gl?", "OpC?"}
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


# a real Messier/NGC/IC object ("NGC 2244", "IC 1805A"), not a cluster member like "NGC 2244 152"
DSO_ID = re.compile(r"^(M|NGC|IC) ?\d+[A-Z]?$")

ID_RANK = [("NAME ", 0), ("M ", 1), ("NGC ", 2), ("IC ", 3), ("* ", 4), ("V* ", 5),
           ("HD ", 6), ("HIP ", 7), ("UGC ", 8), ("PGC ", 9), ("TYC ", 10)]


def rank(i):
    if i.startswith(("M ", "NGC ", "IC ")) and not DSO_ID.match(i):
        return 98  # cluster-member designation: last resort as a name
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


def align_image(img, W, H, cat, rendered, log=print):
    """Transform from FITS display coordinates to display-image pixels.

    cat: [(x, y, obj)] catalog objects in FITS display coordinates. For a supplied PNG
    (which may be cropped, resized, rotated or mirrored) the transform is found by
    matching catalog stars to stars detected in the image.
    """
    ident = (np.eye(2), np.zeros(2))
    if rendered:
        return ident
    stars = sorted(((o["V"] if o["V"] is not None else o["G"], x, y) for x, y, o in cat
                    if (o["V"] is not None or o["G"] is not None) and np.isfinite(x) and np.isfinite(y)
                    and 0 <= x < W and 0 <= y < H), key=lambda r: r[0])
    src = np.array([(x, y) for _, x, y in stars]).reshape(-1, 2)
    dst = register.detect_stars(img)
    res = register.find_transform(src, dst, log=log)
    if res and res[2] >= 8 and res[3] is not None and res[3] < 2.0:
        (M, t), mirrored, n, rms = res
        sc = float(np.sqrt(abs(np.linalg.det(M))))
        rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
        log(f"  aligned image to FITS: {n} stars matched (rms {rms:.2f}px), scale {sc:.3f}, "
            f"rotation {rot:.1f}°{', mirrored' if mirrored else ''}")
        return M, t
    log(f"  warning: could not match stars between the image and the FITS "
        f"({res[2] if res else 0} matches)")
    iw, ih = img.size
    if abs(iw / ih - W / H) < 0.01 * W / H:  # same framing: assume a plain resize
        log("  falling back to a plain resize (same aspect ratio)")
        return np.diag([iw / W, ih / H]), np.array([(iw / W - 1) / 2, (ih / H - 1) / 2])
    raise RuntimeError("Couldn't line up the display image with the FITS (it seems to be cropped and no "
                       "stars could be matched). Try uploading without the image, or with an uncropped one.")


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

        frame_is_png = False  # True when the plate solution is for the PNG itself
        if "CTYPE1" in hdr and "CRVAL1" in hdr:
            wcs = WCS(hdr, naxis=2)
            log("  FITS already plate-solved")
        else:
            from solver import pixel_scale, solve  # astrometry.net fallback
            log("  FITS has no plate solution, solving with astrometry.net…")
            scale_range = None
            if png and starless:  # no stars in the FITS: solve the PNG directly, in its own frame
                solve_data = np.flipud(np.asarray(img.convert("L"), dtype=np.float32) / 255.0)
                s0 = pixel_scale(hdr)
                if s0:  # the PNG may be downsized, so allow coarser pixel scales
                    scale_range = (s0 * 0.8, s0 * 1.25 * max(1.0, W / img.size[0], H / img.size[1]))
                W, H = img.size
                frame_is_png = True
            else:
                solve_data = data
            wcs_hdr = solve(hdr, solve_data, Path(workdir or fit.parent), log=log, scale_range=scale_range)
            if wcs_hdr is None:
                raise RuntimeError("Plate solving failed: could not work out where in the sky this image is.")
            wcs = WCS(wcs_hdr, naxis=2)

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
    dso_rows = tap(f"SELECT {cols},i.id FROM basic b JOIN ident i ON i.oidref=b.oid "
                   f"LEFT JOIN allfluxes f ON b.oid=f.oidref WHERE {region} AND "
                   "(i.id LIKE 'M %' OR i.id LIKE 'NGC %' OR i.id LIKE 'IC %')")
    dsos = list({r["oid"]: r for r in dso_rows if DSO_ID.match(norm_id(r.pop("id")))}.values())
    # the named main object, which may have no M/NGC/IC number (e.g. "NAME Rosette Nebula")
    main_query = (main or str(hdr.get("OBJECT", ""))).strip()
    named = []
    if main_query:
        q = main_query.replace("'", "''")
        named = tap(f"SELECT {cols} FROM ident i JOIN basic b ON b.oid=i.oidref "
                    f"LEFT JOIN allfluxes f ON b.oid=f.oidref WHERE i.id IN ('{q}', 'NAME {q}')")
    objs = {o["oid"]: o for o in stars + dsos + named}
    dso_oids = {o["oid"] for o in dsos + named if not o["otype"].endswith("*") or o["otype"] in CLUSTER_TYPES}
    log(f"  {len(stars)} stars, {len(dsos)} M/NGC/IC objects nearby")

    # positions in "FITS display" coordinates: FITS pixels with the rows flipped to top-down
    def to_display(ra_, dec_):
        x_, y_ = wcs.world_to_pixel_values(ra_, dec_)
        return np.asarray(x_, dtype=float), H - 1 - np.asarray(y_, dtype=float)

    oids = list(objs)
    px, py = to_display([objs[o]["ra"] for o in oids], [objs[o]["dec"] for o in oids])
    T = align_image(img, W, H, [(x, y, objs[o]) for o, x, y in zip(oids, px, py)],
                    rendered=not png or frame_is_png, log=log)
    tx, ty = register.apply(T, np.c_[px, py]).T
    scale_img = float(np.sqrt(abs(np.linalg.det(T[0]))))  # display pixels per FITS pixel
    iw, ih = img.size
    inframe = []
    for o, x, y in zip(oids, tx, ty):
        if np.isfinite(x) and np.isfinite(y) and -2 <= x <= iw + 1 and -2 <= y <= ih + 1:
            objs[o]["x"], objs[o]["y"] = float(x), float(y)
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
    main_oid = next((o["oid"] for o in named if o["oid"] in inframe), None)
    if main_query and main_oid is None:
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
        is_star = o not in dso_oids or (ob["otype"] not in CLUSTER_TYPES
                                        and (ob["otype"].endswith("*") or ob["otype"] in STAR_TYPES))
        m = {"id": mid, "name": name, "aliases": aliases, "x": round(ob["x"], 1), "y": round(ob["y"], 1),
             "type": "Galaxy" if ob["otype"] in GALAXY_TYPES else otypes.get(ob["otype"], ob["otype"]),
             "kind": "star" if is_star else "dso", "mag": round(mag, 2) if mag is not None else None,
             "wiki": wiki,
             "simbad": SIMBAD_WEB + requests.utils.quote(mid),
             "main": o == main_oid}
        if not is_star and ob["galdim_majaxis"]:
            maj = ob["galdim_majaxis"] * 60 / scale_arcsec * scale_img
            mnr = (ob["galdim_minaxis"] or ob["galdim_majaxis"]) * 60 / scale_arcsec * scale_img
            if 12 < maj < 0.6 * math.hypot(*img.size):
                pa = ob["galdim_angle"] or 0
                d = 1 / 60  # a point 1' along the position angle gives the axis direction in the image
                r2 = ob["ra"] + d * math.sin(math.radians(pa)) / math.cos(math.radians(ob["dec"]))
                d2 = ob["dec"] + d * math.cos(math.radians(pa))
                (x1, y1), (x2, y2) = register.apply(T, np.c_[to_display([ob["ra"], r2], [ob["dec"], d2])])
                ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
                m["ellipse"] = [round(maj, 1), round(mnr, 1), round(ang, 1)]
        markers.append(m)

    # merge deep-sky entries that resolve to the same name (e.g. "NAME Heart Nebula" and IC 1805,
    # whose Wikipedia article is "Heart Nebula"): keep the main/linked one, carry over link + aliases
    markers.sort(key=lambda m: (not m["main"], not m["wiki"], m["kind"] != "dso", len(m["name"])))
    by_name, merged = {}, []
    for m in markers:
        k = m["kind"] == "dso" and m["name"].lower()
        if k and k in by_name:
            keep = by_name[k]
            keep["wiki"] = keep["wiki"] or m["wiki"]
            keep["aliases"] = list(dict.fromkeys(keep["aliases"] + [m["id"]] + m["aliases"]))[:6]
            continue
        if k:
            by_name[k] = m
        merged.append(m)
    markers = merged
    # drop duplicates (e.g. a double star and its primary at the same spot): keep the richer entry
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


def reprocess(slugs=None, log=print):
    """Re-tag photos from their saved uploads (<data>/uploads/<slug>/), keeping title and name."""
    uploads = Path(os.environ.get("UPLOADS_DIR", SITE.parent / "uploads"))
    for d in sorted(SITE.glob("*/data.json")):
        j = json.loads(d.read_text())
        slug = j["slug"]
        if slugs and slug not in slugs:
            continue
        src = uploads / slug
        fit = src / (j.get("source", {}).get("fits") or "")
        img = j.get("source", {}).get("image")
        if not fit.is_file():
            log(f"== {slug}: skipped (no saved FITS in {src})")
            continue
        log(f"== {slug}")
        try:
            annotate(fit, src / img if img else None, main=j["title"], title=j["title"], slug=slug,
                     workdir=src / "solve", log=log)
        except Exception as e:  # noqa: BLE001
            log(f"   failed: {e}")


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
    rp = sub.add_parser("reprocess", help="re-tag photos from their saved uploads (after a fix)")
    rp.add_argument("slugs", nargs="*", help="only these photos (default: all)")
    args = ap.parse_args()
    if args.cmd == "add":
        fit, png = find_inputs(args.path, args.png)
        annotate(fit, png, args.main, args.title, args.slug, args.maglim)
    elif args.cmd == "render":
        render_all()
    elif args.cmd == "reprocess":
        reprocess(args.slugs)
    else:
        build_album()


if __name__ == "__main__":
    main()
