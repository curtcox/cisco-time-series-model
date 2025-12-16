# Docker Usage Guide

This guide explains how to build and run the Docker image that serves the Flask CPU forecast app defined in `1.0-preview/cpu_forecast_server.py`.

## Prerequisites

- Docker Desktop (Windows/macOS) or Docker Engine (Linux) 24.0+.
- Internet connectivity to download Python dependencies and the Cisco TSM checkpoint on first launch.

## Build the Image

Run from the repository root:

```bash
docker build -t cisco-cpu-forecast:latest .
```

This copies the repo, installs `requirements.txt`, and sets `cpu_forecast_server.py` as the default entrypoint listening on port `8000`. The Docker image uses Python 3.10.13 to match the verified runtime declared in `.python-version`.

### Use the Published GitHub Container Registry Image

If you prefer not to build locally, every push to `main` automatically publishes an image to the GitHub Container Registry (GHCR). Authenticate with GHCR (a classic PAT with the `read:packages` scope or the standard `GITHUB_TOKEN` in CI) and pull:

```bash
# replace <owner> with the GitHub org/user that hosts this repo
docker pull ghcr.io/<owner>/cisco-cpu-forecast:latest
docker run --rm -it -p 8000:8000 ghcr.io/<owner>/cisco-cpu-forecast:latest
```

> CI jobs also upload a compressed Docker image (`cpu-forecast-server-image.tar.gz`) as a GitHub Action artifact. Download it from the workflow run summary, decompress, and load it with `docker load -i cpu-forecast-server-image.tar.gz`.

## Run the Container

The container exposes port `8000` via the `CPU_FORECAST_SERVER_PORT` environment variable. Map it to your host port with `-p 8000:8000`.

### macOS / Linux (bash/zsh)

```bash
docker run --rm -it -p 8000:8000 cisco-cpu-forecast:latest
```

### Windows PowerShell

```powershell
docker run --rm -it -p 8000:8000 cisco-cpu-forecast:latest
```

> The commands are identical across platforms. Ensure Docker Desktop is running on Windows/macOS. Linux requires the Docker daemon to be active.

## Validate the Server

1. Wait until the container logs show Flask listening on `0.0.0.0:8000`.
2. Open your browser to http://localhost:8000 to load the web UI. Submit the default sample to confirm graphs render.
3. (Optional) Use `curl` to exercise the API endpoint:

   ```bash
   curl -X POST http://localhost:8000/api/forecast \
     -H "Content-Type: application/json" \
     -d '{"use_sample": true, "horizon": 128}'
   ```

   A PNG response (`image/png`) indicates the service is healthy.

4. Stop the container with `Ctrl+C` (Linux/macOS) or `Ctrl+Break` (Windows PowerShell), or by running `docker stop <container_id>` from another terminal.

## Troubleshooting

- **Port already in use:** Change the host side of the mapping, e.g., `-p 8080:8000`, then visit http://localhost:8080.
- **Slow first request:** The model downloads weights on first use; subsequent runs are faster because the checkpoint remains in the Docker layer cache.
- **GPU acceleration:** The default image targets CPU inference. Extend the Dockerfile with NVIDIA CUDA tooling if GPU support is required.
