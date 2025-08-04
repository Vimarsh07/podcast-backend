# main.py

from dotenv import load_dotenv
load_dotenv()

import os
import logging
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from apscheduler.schedulers.background import BackgroundScheduler


import crud
from db import SessionLocal, init_db
from model import User, Podcast as PodcastModel, Episode as EpisodeModel

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── Initialize DB & App ─────────────────────────────────────────────────────
init_db()
app = FastAPI(title="Podcast Summarizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")],
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")



# ─── Dependencies ────────────────────────────────────────────────────────────
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
    username: EmailStr   # front-end should send { "username": "...", "password": "..." }
    password: str

class SubscribeRequest(BaseModel):
    title: str
    feed_url: str

# ─── Auth Routes ─────────────────────────────────────────────────────────────
@app.post("/signup")
def signup(req: SignupRequest, db: Session = Depends(get_db)):
    logger.info(f"▶️  Signing up {req.email}")
    try:
        user = crud.create_user(db, req.email, req.password)
        logger.info(f"✅  User created id={user.id}")
        return user
    except Exception as e:
        logger.error(f"❌  Signup failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    logger.info(f"▶️  Login attempt for {req.username}")
    token = crud.authenticate_user(db, req.username, req.password)
    if not token:
        logger.warning("⚠️  Invalid credentials")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    logger.info("✅  Login successful")
    return {"access_token": token, "token_type": "bearer"}

# ─── Podcast Routes ──────────────────────────────────────────────────────────
@app.get("/podcasts")
def list_podcasts(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    logger.info(f"▶️  Listing podcasts for user_id={user.id}")
    return crud.get_user_podcasts(db, user.id)

@app.get("/podcasts/{podcast_id}")
def get_podcast(
    podcast_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    logger.info(f"▶️  Fetch podcast {podcast_id} for user_id={user.id}")
    if not crud.is_user_subscribed(db, user.id, podcast_id):
        logger.warning("⚠️  Access denied — not subscribed")
        raise HTTPException(status_code=403, detail="Not subscribed")
    pod = db.query(PodcastModel).get(podcast_id)
    if not pod:
        logger.error("❌  Podcast not found")
        raise HTTPException(status_code=404, detail="Not found")
    return pod

@app.post("/podcasts", status_code=201)
def subscribe_podcast(
    req: SubscribeRequest,
    bg: BackgroundTasks,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    logger.info(f"▶️  Subscribing user_id={user.id} to feed={req.feed_url}")
    sub = crud.subscribe_podcast(db, user.id, req.feed_url, title=req.title)
    logger.info(f"✅  Subscribed (podcast_id={sub.podcast_id})")
    bg.add_task(crud.fetch_and_process_latest_async, sub.podcast_id)
    logger.info("✅  Queued first-episode processing")
    return {"podcast_id": sub.podcast_id, "status": "subscribed and queued"}

@app.delete("/podcasts/{podcast_id}", status_code=status.HTTP_204_NO_CONTENT)
def unsubscribe_podcast(
    podcast_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    logger.info(f"▶️  Unsubscribing user_id={user.id} from podcast_id={podcast_id}")
    crud.unsubscribe_podcast(db, user.id, podcast_id)
    logger.info("✅  Unsubscribed")
    return

# ─── Episode Routes ─────────────────────────────────────────────────────────
@app.get("/episodes/{podcast_id}")
def get_episodes(
    podcast_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    logger.info(f"▶️  Listing episodes for podcast_id={podcast_id}")
    if not crud.is_user_subscribed(db, user.id, podcast_id):
        logger.warning("⚠️  Access denied — not subscribed")
        raise HTTPException(status_code=403, detail="Not subscribed")
    return crud.list_episodes(db, podcast_id)

@app.post("/podcasts/{podcast_id}/fetch-latest")
def fetch_latest(
    podcast_id: int,
    bg: BackgroundTasks,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    logger.info(f"▶️  Manual fetch-latest for podcast_id={podcast_id}")
    if not crud.is_user_subscribed(db, user.id, podcast_id):
        logger.warning("⚠️  Access denied — not subscribed")
        raise HTTPException(status_code=403, detail="Not subscribed")
    bg.add_task(crud.fetch_and_process_latest_async, podcast_id)
    logger.info("✅  Queued manual fetch")
    return {"status": "queued"}

# ─── Scheduler for daily polling ─────────────────────────────────────────────
scheduler = BackgroundScheduler()
scheduler.add_job(
    crud.sync_all_podcasts_async,
    trigger="cron",
    hour=0,
    minute=0,
    timezone="UTC"
)
scheduler.start()
logger.info("✅  Scheduler started — daily sync at 00:00 UTC")
