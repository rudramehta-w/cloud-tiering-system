# Cloud Predict

Cloud Predict is an AI-driven hybrid-cloud storage tiering orchestrator. It
uses access history, machine-learning predictions, and storage-cost models to
decide when objects should move between hot and cold storage tiers.

## Project structure

- `src/` - FastAPI application, storage integration, database models, and the
  evaluation engine.
- `alembic/` - Database migration configuration and revisions.
- `config.yaml` - Application pricing and evaluation settings.
- `docker-compose.yml` - Local MinIO storage service.
- `build_*.py` and `generate_presentation_charts.py` - Optional Python tools
  for generating project presentation material.

## Development

The project requires Python 3.12 or newer. Install dependencies with:

```powershell
uv sync
```

Start the local storage service with:

```powershell
docker compose up -d
```

Run the API with:

```powershell
uv run uvicorn src.main:app --reload
```

The generated PowerPoint files and chart assets are intentionally excluded
from version control by `.gitignore`.
