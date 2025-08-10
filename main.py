# ======================== main.py ========================
from dotenv import load_dotenv
load_dotenv()

import os
import logging
import socket
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from apscheduler.schedulers.background import BackgroundScheduler

import crud
from db import SessionLocal, init_db
from model import User, Podcast as PodcastModel, Episode as EpisodeModel

# ─── Logging Configuration ───────────────────────────────────────────────────
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── FastAPI App Setup ───────────────────────────────────────────────────────
app = FastAPI(title="Podcast Summarizer API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# ─── Dependency Injection ────────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    user = crud.get_current_user(db, token)
    if not user:
        logger.warning("Invalid or expired token")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    return user

# ─── Request Schemas ─────────────────────────────────────────────────────────
class SignupRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    username: EmailStr  # expecting {"username": "...", "password": "..."}
    password: str

class SubscribeRequest(BaseModel):
    title: str
    feed_url: str

class TranscribeRequest(BaseModel):
    summary_words: int | None = 800
    force: bool = False

@app.get("/health", include_in_schema=False)
def health():
    """Simple health check endpoint."""
    return {"status": "ok"}

# ─── Startup Events ──────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    # Network diagnostics for database connectivity
    url = os.environ.get("DATABASE_URL", "")
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or 5432

    logger.info(f"🧪 Resolving {host!r}")
    try:
        addrs = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        for family, _, _, _, sockaddr in addrs:
            ip = sockaddr[0]
            fam = "IPv4" if family == socket.AF_INET else "IPv6"
            logger.info(f"  → {fam} addr: {ip}")
            try:
                sock = socket.create_connection((ip, port), timeout=3)
                sock.close()
                logger.info(f"    ✅ {fam} connect OK")
            except Exception as e:
                logger.error(f"    ❌ {fam} connect failed: {e}")
    except Exception as e:
        logger.error(f"  Resolution failed: {e}")

    # Initialize the database (create tables)
    init_db()
    logger.info("✅ Database initialized")

# ─── Authentication Routes ──────────────────────────────────────────────────
@app.post("/signup")
def signup(req: SignupRequest, db: Session = Depends(get_db)):
    logger.info(f"▶️ Signing up {req.email}")
    try:
        user = crud.create_user(db, req.email, req.password)
        logger.info(f"✅ User created id={user.id}")
        return user
    except Exception as e:
        logger.error(f"❌ Signup failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    logger.info(f"▶️ Login attempt for {req.username}")
    token = crud.authenticate_user(db, req.username, req.password)
    if not token:
        logger.warning("⚠️ Invalid credentials")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    logger.info("✅ Login successful")
    return {"access_token": token, "token_type": "bearer"}

# ─── Podcast Routes ──────────────────────────────────────────────────────────
@app.get("/podcasts")
def list_podcasts(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    logger.info(f"▶️ Listing podcasts for user_id={user.id}")
    return crud.get_user_podcasts(db, user.id)

@app.get("/podcasts/{podcast_id}")
def get_podcast(
    podcast_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    logger.info(f"▶️ Fetch podcast {podcast_id} for user_id={user.id}")
    if not crud.is_user_subscribed(db, user.id, podcast_id):
        logger.warning("⚠️ Access denied — not subscribed")
        raise HTTPException(status_code=403, detail="Not subscribed")
    pod = db.query(PodcastModel).get(podcast_id)
    if not pod:
        logger.error("❌ Podcast not found")
        raise HTTPException(status_code=404, detail="Not found")
    return pod

@app.post("/podcasts", status_code=201)
def subscribe_podcast(
    req: SubscribeRequest,
    bg: BackgroundTasks,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    logger.info(f"▶️ Subscribing user_id={user.id} to feed={req.feed_url}")
    sub = crud.subscribe_podcast(db, user.id, req.feed_url, title=req.title)
    logger.info(f"✅ Subscribed (podcast_id={sub.podcast_id})")
    # Queue metadata-only ingest so UI can show latest 10 episodes to pick from
    bg.add_task(crud.fetch_latest_metadata_async, sub.podcast_id, 10)
    logger.info("✅ Queued metadata ingest")
    return {"podcast_id": sub.podcast_id, "status": "subscribed and queued"}

@app.delete("/podcasts/{podcast_id}", status_code=status.HTTP_204_NO_CONTENT)
def unsubscribe_podcast(
    podcast_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    logger.info(f"▶️ Unsubscribing user_id={user.id} from podcast_id={podcast_id}")
    crud.unsubscribe_podcast(db, user.id, podcast_id)
    logger.info("✅ Unsubscribed")

# ─── Episode Routes ─────────────────────────────────────────────────────────
@app.get("/episodes/{podcast_id}")
def get_episodes(
    podcast_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    logger.info(f"▶️ Listing episodes for podcast_id={podcast_id}")
    if not crud.is_user_subscribed(db, user.id, podcast_id):
        logger.warning("⚠️ Access denied — not subscribed")
        raise HTTPException(status_code=403, detail="Not subscribed")
    return crud.list_episodes(db, podcast_id)

@app.post("/podcasts/{podcast_id}/fetch-latest")
def fetch_latest(
    podcast_id: int,
    bg: BackgroundTasks,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    limit: int = 10
):
    logger.info(f"▶️ Manual fetch-latest for podcast_id={podcast_id} (limit={limit})")
    if not crud.is_user_subscribed(db, user.id, podcast_id):
        logger.warning("⚠️ Access denied — not subscribed")
        raise HTTPException(status_code=403, detail="Not subscribed")
    # Queue metadata-only ingest (no transcription)
    bg.add_task(crud.fetch_latest_metadata_async, podcast_id, limit)
    logger.info("✅ Queued metadata ingest")
    return {"status": "queued", "limit": limit}

@app.post("/episodes/{episode_id}/transcribe-and-summarize")
def transcribe_and_summarize(
    episode_id: int,
    body: TranscribeRequest,
    bg: BackgroundTasks,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # Make sure the user is subscribed to the parent podcast
    ep = db.query(EpisodeModel).get(episode_id)
    if not ep:
        raise HTTPException(status_code=404, detail="Episode not found")
    if not crud.is_user_subscribed(db, user.id, ep.podcast_id):
        raise HTTPException(status_code=403, detail="Not subscribed to this podcast")

    # If not forcing and it's already completed, short-circuit
    if not body.force and getattr(ep, "transcript", None) and getattr(ep, "summary", None):
        return {"message": "Already completed", "episode_id": episode_id}

    # Queue the job; CRUD will update transcript/summary/status on the same row
    bg.add_task(
        crud.transcribe_and_summarize_episode_async,
        episode_id,
        (body.summary_words or 800)
    )
    return {"message": "Queued", "episode_id": episode_id}

# ─── Scheduler for Daily Sync ────────────────────────────────────────────────
scheduler = BackgroundScheduler()
scheduler.add_job(
    crud.sync_all_podcasts_metadata_async,  # << updated function
    trigger="cron",
    hour=0,
    minute=0,
    timezone="UTC"
)
scheduler.start()
logger.info("✅ Scheduler started — daily metadata sync at 00:00 UTC")

# ─── Application Entry Point ─────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
