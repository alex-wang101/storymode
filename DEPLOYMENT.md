# Docker Deployment

This repo can run in Docker in two ways:

- API server: starts FastAPI on port `8000`.
- CLI runner: runs `python main.py ...` inside the same image.

The image is CPU-oriented. On an Apple Silicon Mac, Docker containers do not get
native MPS acceleration, so expect CPU runtime to be slower than running the
repo directly on macOS with MPS. On Linux servers with Docker, this setup is the
most portable baseline.

## Prerequisites

Install Docker Desktop on macOS/Windows, or Docker Engine plus Docker Compose on
Linux.

Check that Docker works:

```bash
docker --version
docker compose version
```

From the repo root, build the image:

```bash
docker compose build
```

The first build is large because it installs PyTorch, Transformers, OCR,
OpenCV, FastAPI, and the local/cloud VLM dependencies.

## Files Docker Uses

- `Dockerfile`: builds the Python runtime and installs system/Python deps.
- `.dockerignore`: keeps local caches, videos, and experiment outputs out of
  the image.
- `docker-compose.yml`: convenient local deployment with persistent volumes.

The compose file mounts:

- `./data:/app/data` so local videos, references, and job cache persist.
- `./experiments:/app/experiments` so CLI output files persist.
- `videofind-model-cache:/home/app/.cache` so Hugging Face, Torch, EasyOCR,
  and other downloaded model files survive container rebuilds/restarts.

## Run The API Server

Start the service:

```bash
docker compose up
```

Open the interactive API docs:

```text
http://localhost:8000/docs
```

Run it in the background:

```bash
docker compose up -d
docker compose logs -f videofind
```

Stop it:

```bash
docker compose down
```

### Optional API Token

By default the API accepts requests without auth. To require a bearer token,
create a `.env` file next to `docker-compose.yml`:

```bash
VIDEOFIND_TOKEN=replace-with-a-long-random-token
```

Restart:

```bash
docker compose up -d
```

Requests must then include:

```text
Authorization: Bearer replace-with-a-long-random-token
```

### API Example

The API expects a YouTube video URL and an HTTPS reference image URL. For local
VLM reference generation:

```bash
curl -X POST http://localhost:8000/jobs/local \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "reference_image_url": "https://example.com/product.jpg",
    "product_name_hint": "optional product name"
  }'
```

With `VIDEOFIND_TOKEN` set:

```bash
curl -X POST http://localhost:8000/jobs/local \
  -H "Authorization: Bearer replace-with-a-long-random-token" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "reference_image_url": "https://example.com/product.jpg"
  }'
```

Cloud reference generation uses `/jobs/cloud` and requires an `api_key` in the
JSON body. The app routes by key prefix in `api/reference_gen.py`.

## Run CLI Commands

Use the image without starting the API:

```bash
docker compose run --rm videofind python main.py detect-sponsors
docker compose run --rm videofind python main.py detect-sponsors --url "https://www.youtube.com/watch?v=VIDEO_ID"
docker compose run --rm videofind python main.py detect-text --output experiments/stage2/text.json
docker compose run --rm videofind python main.py detect-object --output experiments/stage2/object.json
```

For bundled-video runs, place the expected MP4 files on the host first:

```text
data/videos/zbiotics-bacon.mp4
data/videos/zbiotics-pickles.mp4
```

Because `./data` is mounted into `/app/data`, files you place in local
`data/videos/` are visible inside the container.

## Run Without Compose

Build:

```bash
docker build -t videofind:local .
```

Run the API:

```bash
docker run --rm -p 8000:8000 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/experiments:/app/experiments" \
  -v videofind-model-cache:/home/app/.cache \
  videofind:local
```

Run a CLI command:

```bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/experiments:/app/experiments" \
  -v videofind-model-cache:/home/app/.cache \
  videofind:local python main.py detect-sponsors
```

## First Run Expectations

The first real job downloads model weights into the Docker volume. That can take
several minutes and multiple GB of disk. Later runs reuse the cache.

Keep at least 20 GB free for:

- Docker image layers.
- model cache volume.
- downloaded videos and intermediate job cache under `data/cache/`.
- CLI outputs under `experiments/`.

## Maintenance

Rebuild after dependency or Dockerfile changes:

```bash
docker compose build
docker compose up -d
```

Delete generated job/video cache:

```bash
rm -rf data/cache/*
```

Delete downloaded model cache too:

```bash
docker compose down -v
```

That removes the named Docker volumes, so the next run will download models
again.

## Common Problems

`ModuleNotFoundError` after changing dependencies:

```bash
docker compose build --no-cache
```

`no space left on device`: clear `data/cache/`, old Docker images, or the model
volume with `docker compose down -v`.

Very slow inference: the container is using CPU. For best Apple Silicon speed,
run the Python app directly on macOS so PyTorch can use MPS.

Video download failures: `yt-dlp` depends on the source site and may fail for
private, age-gated, live, region-blocked, or format-restricted videos.
