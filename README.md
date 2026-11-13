# Astro Album

An album for smart-telescope astrophotos (built around the DWARF 3). Upload a FITS file and
every catalogued star and deep-sky object in the photo is marked. Hover over a marker for
its name, type and magnitude, plus links to Wikipedia and SIMBAD.

## How it works

1. **Where is the photo pointing?** The FITS header's plate solution (WCS), e.g. from Siril,
   maps every pixel to sky coordinates. If the FITS isn't plate-solved, the server solves it
   with astrometry.net (star index files are downloaded once, on first use).
2. **What's in the frame?** SIMBAD is queried for stars down to magnitude 11.5 and for every
   Messier/NGC/IC object in the field, then each is projected onto the image.
3. **Wikipedia links** come from Wikidata (SIMBAD ID → English Wikipedia article).
4. The display image is your processed PNG/JPG if you supply one, otherwise it's rendered
   (and auto-stretched if linear) from the FITS.

## Run on a server (Docker)

```bash
docker compose up -d        # uses ghcr.io/amcarvalho/astro-album:latest
```

Open `http://<server>:8088` and use **+ Add photo**. Everything persistent lives in `./data`:

| Path | What |
|---|---|
| `data/site/` | the generated album (static HTML + images) |
| `data/uploads/<photo>/` | the original uploaded FITS (and image) |
| `data/astrometry-index/` | plate-solving star indexes (~200 MB, only if needed) |

Update to the latest image: `docker compose pull && docker compose up -d`.

The image is built for amd64 and arm64 by GitHub Actions on every push to `main`
(`.github/workflows/docker.yml`).

## Command line

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python album.py add "path/to/folder-with-fit-and-png" --main "Andromeda Galaxy"
.venv/bin/python album.py render    # re-apply templates/ to every photo
```

Run the server locally: `DATA_DIR=./data .venv/bin/uvicorn server:app --port 8088`.
