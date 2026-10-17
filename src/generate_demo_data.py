import os
import random
import string
import numpy as np
from datetime import datetime, timedelta
from src.database import SessionLocal
from src.models import FileMetadata, AccessEvent

def generate_id(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def generate_demo_data():
    print("Generating 60 days of historical file data for the seminar...")
    db = SessionLocal()
    
    # Clear existing data so demo is clean
    db.query(AccessEvent).delete()
    db.query(FileMetadata).delete()
    db.commit()

    NUM_FILES = 200
    DAYS = 60
    
    now = datetime.utcnow()
    start_date = now - timedelta(days=DAYS)
    
    files = []
    # Create 200 files
    for i in range(NUM_FILES):
        file_id = f"demo_file_{i}_{generate_id(4)}.dat"
        # Sizes between 10MB and 2GB
        size_bytes = random.randint(10 * 1024**2, 2 * 1024**3) 
        
        # Files created at random times over the 60 days
        created_at = start_date + timedelta(days=random.randint(0, DAYS-1), hours=random.randint(0, 23))
        
        fm = FileMetadata(id=file_id, size_bytes=size_bytes, tier='hot', created_at=created_at)
        db.add(fm)
        
        # Upload event
        ev = AccessEvent(file_id=file_id, event_type='upload', timestamp=created_at)
        db.add(ev)
        
        files.append((file_id, created_at))
        
    db.commit()
    
    # Generate Zipfian accesses (some files accessed a lot, long tail accessed rarely)
    ranks = np.arange(1, NUM_FILES + 1)
    probabilities = 1.0 / np.power(ranks, 1.5)
    probabilities /= np.sum(probabilities)
    
    # 3000 access events over 60 days
    NUM_EVENTS = 3000
    for _ in range(NUM_EVENTS):
        # Pick a file based on zipf distribution
        file_idx = np.random.choice(NUM_FILES, p=probabilities)
        file_id, created_at = files[file_idx]
        
        # Access must happen after creation
        age_days = (now - created_at).days
        if age_days <= 0:
            continue
            
        access_time = created_at + timedelta(days=random.uniform(0, age_days))
        
        ev = AccessEvent(file_id=file_id, event_type='access', timestamp=access_time)
        db.add(ev)
        
    db.commit()
    print(f"Success! Inserted {NUM_FILES} files and ~{NUM_EVENTS} access events.")
    print("The system is now populated with rich data for your presentation!")

if __name__ == "__main__":
    generate_demo_data()
