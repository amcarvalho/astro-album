FROM python:3.12-slim-bookworm

# astrometry.net provides solve-field, used when an uploaded FITS isn't plate-solved yet
RUN apt-get update \
 && apt-get install -y --no-install-recommends astrometry.net \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY album.py register.py solver.py server.py ./
COPY templates ./templates

ENV DATA_DIR=/data PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
