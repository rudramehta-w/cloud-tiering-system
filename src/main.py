import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime

from src.config import get_settings
from src.database import get_db, engine
from src.models import FileMetadata, AccessEvent
from src.storage import storage
from src.engine import evaluate_and_migrate
from sqlalchemy.orm import Session

settings = get_settings()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # APScheduler start
    scheduler.add_job(evaluate_and_migrate, 'interval', minutes=60)
    scheduler.start()
    
    # Initialize AWS S3 Buckets if they don't exist
    async with storage._client() as s3:
        try:
            await s3.create_bucket(Bucket=settings.storage.hot_bucket)
            await s3.create_bucket(Bucket=settings.storage.cold_bucket)
        except Exception:
            pass
            
    yield
    scheduler.shutdown()

app = FastAPI(
    title="Cloud Predict Storage API",
    description="Professional Grade Hybrid-Cloud Storage Tiering Orchestrator",
    version="2.0.0",
    lifespan=lifespan
)

class UploadRequest(BaseModel):
    file_id: str = Field(..., description="Unique identifier for the file")
    size_bytes: int = Field(..., description="Size of the file in bytes", gt=0)

class PresignedURLResponse(BaseModel):
    url: str
    message: str

@app.post("/api/v1/upload", response_model=PresignedURLResponse, tags=["Storage"])
async def request_upload(request: UploadRequest, db: Session = Depends(get_db), api_key: str = Depends(verify_api_key)):
    """ Request a presigned URL to upload a file directly to the hot storage tier. """
    url = await storage.generate_presigned_url('put_object', settings.storage.hot_bucket, request.file_id)
    
    file_meta = FileMetadata(id=request.file_id, size_bytes=request.size_bytes, tier='hot')
    db.merge(file_meta)
    
    event = AccessEvent(file_id=request.file_id, event_type='upload')
    db.add(event)
    db.commit()
    
    return {"url": url, "message": "Ready for direct client upload"}

@app.get("/api/v1/download/{file_id}", response_model=PresignedURLResponse, tags=["Storage"])
async def request_download(file_id: str, db: Session = Depends(get_db), api_key: str = Depends(verify_api_key)):
    """ Request a presigned URL to download a file from whichever tier it currently resides in. """
    file_meta = db.query(FileMetadata).filter(FileMetadata.id == file_id).first()
    if not file_meta:
        raise HTTPException(status_code=404, detail="File not found")
        
    bucket = settings.storage.hot_bucket if file_meta.tier == 'hot' else settings.storage.cold_bucket
    url = await storage.generate_presigned_url('get_object', bucket, file_id)
    
    event = AccessEvent(file_id=file_id, event_type='access')
    db.add(event)
    db.commit()
    
    return {"url": url, "message": f"File downloading from {file_meta.tier} tier"}

from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from fastapi import Request

app.mount("/static", StaticFiles(directory="src/static"), name="static")
templates = Jinja2Templates(directory="src/templates")

@app.get("/api/v1/stats", tags=["Monitoring"])
def get_stats(db: Session = Depends(get_db)):
    hot_bytes = db.query(func.sum(FileMetadata.size_bytes)).filter(FileMetadata.tier == 'hot').scalar() or 0
    cold_bytes = db.query(func.sum(FileMetadata.size_bytes)).filter(FileMetadata.tier == 'cold').scalar() or 0
    total_files = db.query(func.count(FileMetadata.id)).scalar() or 0
    
    hot_gb = hot_bytes / (1024**3)
    cold_gb = cold_bytes / (1024**3)
    
    # Calculate current savings per month just based on cold tier volume
    savings = cold_gb * (settings.pricing.hot_storage_per_gb_month - settings.pricing.cold_storage_per_gb_month)
    
    recent_files = db.query(FileMetadata).order_by(FileMetadata.created_at.desc()).limit(10).all()
    
    return {
        "hot_gb": hot_gb,
        "cold_gb": cold_gb,
        "total_files": total_files,
        "savings_usd": savings,
        "recent_files": [{"id": f.id, "size_bytes": f.size_bytes, "tier": f.tier} for f in recent_files]
    }

@app.post("/api/v1/trigger-evaluation", tags=["Monitoring"])
async def trigger_evaluation():
    """Manually trigger the XGBoost evaluation cycle for the live demo."""
    import asyncio
    asyncio.create_task(evaluate_and_migrate())
    return {"status": "Evaluation started in background"}

@app.get("/dashboard", response_class=HTMLResponse, tags=["UI"])
def serve_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
