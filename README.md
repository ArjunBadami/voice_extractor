Voice Recording & Extraction (Skeleton)

Minimal FastAPI app to record audio in the browser, upload to the backend, and persist metadata.

Quickstart
1) Create venv and install deps (Windows Git Bash):
   - python -m venv .venv
   - source .venv/Scripts/activate
   - pip install -r requirements.txt
2) Run:
   - uvicorn main:app --reload
3) Open:
   - http://127.0.0.1:8000
   - Click Start, then Stop & Upload

Endpoints
- GET /healthz
- GET /                (serves static/index.html)
- POST /upload         (saves audio; returns id and status)
- GET /recordings
- GET /recordings/{id}

Project Structure
- main.py              (FastAPI app and endpoints)
- static/index.html    (minimal recorder UI using MediaRecorder)
- data/audio/          (uploaded audio files; gitignored)
- app.db               (SQLite database; gitignored)

Notes
- SQLite is used for simplicity (single file app.db).
- Browser records WebM/Opus; backend accepts multipart/form-data.