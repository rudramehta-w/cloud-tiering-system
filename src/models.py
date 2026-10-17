from datetime import datetime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime

class Base(DeclarativeBase):
    pass

class FileMetadata(Base):
    __tablename__ = "files"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    tier: Mapped[str] = mapped_column(String, default="hot")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

class AccessEvent(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_id: Mapped[str] = mapped_column(String, index=True)
    event_type: Mapped[str] = mapped_column(String) # 'upload' or 'access'
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
