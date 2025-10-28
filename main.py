import os
import re
import json
import uuid
import datetime as dt
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import os


# Config
DATA_DIR = Path("data/audio")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_URL = "sqlite:///app.db"

USE_FAKE_TRANSCRIBER = False

# DB setup (SQLAlchemy)
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Recording(Base):
    __tablename__ = "recordings"
    id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)
    audio_path = Column(Text, nullable=False)
    transcript = Column(Text, nullable=True)
    extraction_json = Column(Text, nullable=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic models
class RecordingSummary(BaseModel):
    id: str
    created_at: dt.datetime

class RecordingDetail(BaseModel):
    id: str
    created_at: dt.datetime
    audio_path: str
    transcript: Optional[str] = None
    extraction_json: Optional[dict] = None


def extract_entities(text: str) -> dict:
    return {
    }


def transcribe_audio(file_path: str) -> str:
    if USE_FAKE_TRANSCRIBER:
        return "Sample transcript."
    return ""

# FastAPI app
app = FastAPI(title="Voice Recording & Data Extraction")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/", response_class=FileResponse)
def index():
    return FileResponse("static/index.html")

@app.post("/upload")
def upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Save file
    rec_id = str(uuid.uuid4())
    ext = ".webm"
    out_path = DATA_DIR / f"{rec_id}{ext}"
    with open(out_path, "wb") as f:
        f.write(file.file.read())

    transcript = transcribe_audio(str(out_path))

    extraction = extract_entities(transcript)

    # Persist
    row = Recording(
        id=rec_id,
        created_at=dt.datetime.utcnow(),
        audio_path=str(out_path),
        transcript=transcript,
        extraction_json=json.dumps(extraction),
    )
    db.add(row)
    db.commit()

    return {
        "id": rec_id,
        "transcript": transcript,
        "extraction": extraction,
        "audio_path": str(out_path),
    }

@app.get("/recordings", response_model=List[RecordingSummary])
def list_recordings(db: Session = Depends(get_db)):
    rows = db.query(Recording).order_by(Recording.created_at.desc()).all()
    return [{"id": r.id, "created_at": r.created_at} for r in rows]

@app.get("/recordings/{rec_id}", response_model=RecordingDetail)
def get_recording(rec_id: str, db: Session = Depends(get_db)):
    row = db.query(Recording).filter(Recording.id == rec_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    return RecordingDetail(
        id=row.id,
        created_at=row.created_at,
        audio_path=row.audio_path,
        transcript=row.transcript,
        extraction_json=json.loads(row.extraction_json) if row.extraction_json else None,
    )
