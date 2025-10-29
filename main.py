import os
import re
import json
import uuid
import datetime as dt
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from openai import OpenAI

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client=OpenAI(api_key=OPENAI_API_KEY)

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



#LLM BASED EXTRACTION START
# ---------- Pydantic schema (what you want back) ----------
class Amount(BaseModel):
    value: Optional[float] = Field(None, description="numeric amount")
    currency: Optional[str] = Field(None, description='e.g. "USD", "INR"')

class LLMExtraction(BaseModel):
    amount: Optional[List[Amount]] = None          # one primary amount if present
    date: Optional[str] = None               # ISO date "YYYY-MM-DD" if possible
    action: Optional[str] = None             # short imperative or intent
    notes: Optional[str] = None              # anything extra
    emails: List[str] = []                   # zero or more emails
    phones: List[str] = []                   # zero or more phones
    risk_tolerance: str=None
    sentiment: str=None

# ---------- helper: pull JSON from any model reply ----------
_JSON_RE = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL)

def _first_json_block(s: str) -> str:
    m = _JSON_RE.search(s)
    if m:
        return m.group(1)
    # fall back: try raw parse
    if s.strip().startswith("{"):
        return s.strip()
    raise ValueError("No JSON object found in model response")

# ---------- the LLM extractor ----------
SYSTEM = (
    "You extract structured data from text.\n"
    "Return STRICT JSON ONLY that matches this schema:\n"
    "{"
    '"amount":[{"value":float|null,"currency":str|null}],'
    '"date":str|null,'
    '"action":str|null,'
    '"notes":str|null,'
    '"emails":[str],'
    '"phones":[str],'
    '"risk_tolerance":str'
    '"sentiment":str'
    "}\n"
    "No explanations, no extra keys."
)

USER_TMPL = (
    "Text:\n\"\"\"\n{transcript}\n\"\"\"\n\n"
    "Instructions:\n"
    "1) If an amount exists, set amount.value (number) and amount.currency (e.g., USD, INR) if inferable. Even if the currency is not known, and only the value is mentioned, assume USD.\n"
    "2) Prefer an explicit date; format as YYYY-MM-DD if possible.\n"
    "3) Derive a short action if implied (e.g., 'pay Alex').\n"
    "4) Put remaining info in notes.\n"
    "5) Extract any emails and phone numbers.\n"
    "6) Risk tolerance of the speaker, based on the contents of their voice note. Asign their risk tolerance into categories: 1- Very Low, 2- Low, 3- Medium, 4-High, 5-Very High.\n"
    "7) Gauge the sentiment of the speaker. Give them a sentiment or optimistic score from 0-100 where 0 is very pessimistic and 100 is very optimistic.\n"
    "Return JSON ONLY."
)

def extract_with_llm(transcript: str) -> dict:
    """
    Calls an LLM to produce strict-JSON extraction, validates with Pydantic,
    retries once with a stricter reminder if needed.
    """
    def _call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # any GPT that you have access to is fine
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content

    prompt = USER_TMPL.format(transcript=transcript)
    for attempt in (1, 2):
        raw = _call(prompt if attempt == 1 else prompt + "\n\nReturn JSON only, no prose.")
        try:
            data = json.loads(_first_json_block(raw))
            obj = LLMExtraction(**data)        # Pydantic validation
            return obj.model_dump()
        except (ValueError, ValidationError, json.JSONDecodeError):
            if attempt == 2:
                # final fallback: empty but schema-correct object
                return LLMExtraction().model_dump()

# ---------- example hybrid use in your /upload flow ----------
# 1) keep your regex-based extract_entities(text)
# 2) merge: prefer deterministic results, then LLM fills gaps

def extract_llm(transcript: str) -> dict:
    #det = extract_entities(transcript)          # your regex results
    llm = extract_with_llm(transcript)          # schema-validated LLM
    # simple merge: keep deterministic hits; fill missing from LLM
    out = {
        #"emails": list({*llm.get("emails", [])}),
        #"phones": list({*llm.get("phones", [])}),
        "llm": llm,  # keep the structured extraction too
    }
    # if deterministic found an amount like "$135", also set llm.amount if empty
    if not llm.get("amount"):
        out["llm"]["amount"] = {"value": None, "currency": None}
    return out
#END


def extract_entities(text: str) -> dict:
    return {
    }


def transcribe_audio(file_path: str) -> str:
    if USE_FAKE_TRANSCRIBER:
        return "Sample transcript."
    if not OPENAI_API_KEY:
        return "API KEY NOT SET"

    with open(file_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=f
        )
    
    if hasattr(transcription, "text"):
        return transcription.text.strip()
    else:
        return str(transcription).strip()


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


    transcript = "Hey Alex, yeah, things have been okay. Busy with work, and honestly, I’ve been watching the markets a bit too much lately. The volatility’s making me nervous — it feels like every other day the headlines are predicting something new. I know last time we talked I said I was comfortable with a moderate risk profile, around sixty percent equities and forty percent fixed income, but lately I’ve been thinking maybe I should dial that back a bit. I’m not saying I want to get completely out of stocks, but I’d feel better if a bit more of the portfolio was in bonds or even cash. Right now, my overall portfolio is about one point one million. The Fidelity rollover IRA is around six hundred fifty thousand, my 401(k) at work is sitting at about one hundred fifty thousand, and the joint brokerage account that Sarah and I have is around two hundred seventy-five thousand. Then there’s roughly seventy-five thousand in cash from the inheritance from my dad — that’s still just sitting in our Chase savings account earning basically nothing. I haven’t decided whether to invest that or just leave it as an emergency fund. I’m about ten years out from retirement — I’ll be sixty-five in 2035 — and I really don’t want to see big swings in the account at this stage. Sarah keeps asking whether we’re still on track for the lake cabin idea, and I told her probably, but I’d like to double-check that with you. Our daughter Emma is starting college next year, and tuition is definitely going to be a hit — around fifty grand a year, I think. I was reading some articles about treasury yields going up, and I’m not sure if that changes anything for us. Should we be looking at shorter-term Treasuries or municipal bonds, or just stay the course? And on the equity side, that tech ETF — QQQ — has been all over the place. It was great last year, but now it’s down maybe fifteen percent from the peak. The S&P index fund in the IRA is also down around five percent year to date. I guess what I’m really asking is: should we make some adjustments now or just ride it out? I’d rather not overreact, but I’m definitely feeling less risk-tolerant than I was a year ago. Maybe shift ten percent more into bonds or cash equivalents. Also, Sarah mentioned maybe we should start thinking about setting up a trust, just to make things easier down the line. Is that something we can talk about next time? And while we’re at it, could we maybe set up a follow-up meeting next week to go through the portfolio and run some scenarios? Maybe Wednesday afternoon if that works for you. I’d just like to see how things would look if we rebalanced a bit and put that inheritance money to work in a lower-risk way. That’s basically what’s been on my mind — just trying to make sure we’re being smart and not taking unnecessary risks this close to retirement."
    #transcript = transcribe_audio(str(out_path))


    extraction = extract_llm(transcript)

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
