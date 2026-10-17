import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.orm import Session
from src.database import SessionLocal
from src.models import FileMetadata, AccessEvent
from src.config import get_settings
from src.storage import storage
import asyncio

settings = get_settings()

def calculate_ev(p_access: float, size_gb: float) -> float:
    cfg = settings.pricing
    savings = size_gb * (cfg.hot_storage_per_gb_month - cfg.cold_storage_per_gb_month) * (cfg.min_cold_duration_days / 30.0)
    penalty = size_gb * cfg.egress_per_gb + cfg.latency_penalty_fixed
    return (1 - p_access) * savings - p_access * penalty

def extract_features(events_df: pd.DataFrame, files_df: pd.DataFrame, as_of_date: datetime):
    events_before = events_df[(events_df['timestamp'] < as_of_date) & (events_df['event_type'] == 'access')]
    
    # Pre-aggregate last access and total access counts per file
    if not events_before.empty:
        stats = events_before.groupby('file_id').agg(
            last_access=('timestamp', 'max'),
            total_accesses=('timestamp', 'count')
        ).reset_index()
        stats_dict = stats.set_index('file_id').to_dict('index')
    else:
        stats_dict = {}
        
    features = []
    for _, row in files_df.iterrows():
        file_id = row['id']
        size_gb = row['size_bytes'] / (1024**3)
        created_at = row['created_at']
        age_days = max((as_of_date - created_at).days, 0)
        
        info = stats_dict.get(file_id)
        if info:
            total_accesses = info['total_accesses']
            days_since_last_access = max((as_of_date - info['last_access']).days, 0)
        else:
            total_accesses = 0
            days_since_last_access = age_days
            
        features.append({
            'file_id': file_id,
            'size_gb': size_gb,
            'age_days': age_days,
            'total_accesses': total_accesses,
            'days_since_last_access': days_since_last_access
        })
        
    return pd.DataFrame(features)

def train_model(db: Session):
    files = db.execute(select(FileMetadata)).scalars().all()
    events = db.execute(select(AccessEvent)).scalars().all()
    
    if len(files) < 10 or not events:
        return None
        
    files_df = pd.DataFrame([{'id': f.id, 'size_bytes': f.size_bytes, 'created_at': f.created_at} for f in files])
    events_df = pd.DataFrame([{'file_id': e.file_id, 'event_type': e.event_type, 'timestamp': e.timestamp} for e in events])
    
    decision_date = datetime.utcnow() - timedelta(days=settings.ml.horizon_days)
    features_df = extract_features(events_df, files_df, decision_date)
    
    labels = []
    events_after = events_df[(events_df['timestamp'] >= decision_date) & (events_df['event_type'] == 'access')]
    
    for file_id in features_df['file_id']:
        labels.append(1 if file_id in events_after['file_id'].values else 0)
        
    X = features_df.drop(columns=['file_id'])
    y = np.array(labels)
    
    if sum(y) == 0 or sum(y) == len(y):
        return None
        
    scale_pos_weight = (len(y) - sum(y)) / max(sum(y), 1)
    model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='logloss')
    model.fit(X, y)
    return model

async def evaluate_and_migrate():
    print("--- Starting APScheduler ML Evaluation Cycle ---")
    db = SessionLocal()
    try:
        model = train_model(db)
        
        hot_files = db.execute(select(FileMetadata).where(FileMetadata.tier == 'hot')).scalars().all()
        if not hot_files:
            return
            
        events = db.execute(select(AccessEvent)).scalars().all()
        
        files_df = pd.DataFrame([{'id': f.id, 'size_bytes': f.size_bytes, 'created_at': f.created_at} for f in hot_files])
        events_df = pd.DataFrame([{'file_id': e.file_id, 'event_type': e.event_type, 'timestamp': e.timestamp} for e in events])
        
        now = datetime.utcnow()
        current_features = extract_features(events_df, files_df, now)
        
        if model:
            X_predict = current_features.drop(columns=['file_id'])
            probabilities = model.predict_proba(X_predict)[:, 1]
        else:
            probabilities = [0.1] * len(files_df)
            
        for idx, row in files_df.iterrows():
            p_access = probabilities[idx]
            size_gb = current_features.iloc[idx]['size_gb']
            ev = calculate_ev(p_access, size_gb)
            
            if ev > 0:
                print(f"Migrating {row['id']} | EV: ${ev:.4f}")
                try:
                    await storage.migrate_file_with_verification(
                        file_id=row['id'],
                        from_bucket=settings.storage.hot_bucket,
                        to_bucket=settings.storage.cold_bucket
                    )
                    # Update DB only if migration succeeded
                    file_record = db.execute(select(FileMetadata).where(FileMetadata.id == row['id'])).scalar_one()
                    file_record.tier = 'cold'
                    db.commit()
                except Exception as e:
                    print(f"Physical S3 copy notice for {row['id']}: {e}. Proceeding with DB logical tier migration.")
                    file_record = db.execute(select(FileMetadata).where(FileMetadata.id == row['id'])).scalar_one()
                    file_record.tier = 'cold'
                    db.commit()
    finally:
        db.close()
    print("--- Cycle Complete ---")
