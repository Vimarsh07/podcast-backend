# main.py

from dotenv import load_dotenv
load_dotenv()  # Load DATABASE_URL, SECRET_KEY, etc. from .env

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

import crud
from db import SessionLocal, init_db
from model import Episode, User

class SubscribeRequest(BaseModel):
    feed_url: str

# 1) Create all tables if they don't exist
init_db()

# 2) Instantiate FastAPI
app = FastAPI(title="Podcast Summarizer API")

# 3) Enable CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4) Dependency: provide a database session for each request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 5) OAuth2 scheme for extracting Bearer token from Authorization header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# 6) Pydantic models for request bodies
class SignupRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# 7) Dependency: get the current authenticated user
def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    user = crud.get_current_user(db, token)  # raises 401 on invalid
    return user

# --- Authentication Endpoints ---

@app.post("/signup")
def signup(req: SignupRequest, db: Session = Depends(get_db)):
    """
    Create a new user. Expects JSON body:
      { "email": "...", "password": "..." }
    """
    return crud.create_user(db, req.email, req.password)

@app.post("/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate user. Expects JSON body:
      { "email": "...", "password": "..." }
    Returns a JWT token on success.
    """
    token = crud.authenticate_user(db, req.email, req.password)
    if not token:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"access_token": token, "token_type": "bearer"}

# --- Podcast Subscription Endpoints ---

@app.get("/podcasts")
def list_podcasts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List all podcasts the current user is subscribed to.
    """
    return crud.get_user_podcasts(db, current_user.id)

@app.post("/podcasts")
def subscribe_podcast(
    req: SubscribeRequest,  # now reads JSON { "feed_url": "…" }
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Subscribe the current user to a podcast by its RSS feed URL.
    Expects JSON body: { "feed_url": "https://…" }
    """
    return crud.subscribe_podcast(db, current_user.id, req.feed_url)

# --- Episode Retrieval Endpoint ---

@app.get("/episodes/{podcast_id}")
def get_episodes(
    podcast_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Return all episodes (with summary/transcript) for a podcast.
    """
    if not crud.is_user_subscribed(db, current_user.id, podcast_id):
        raise HTTPException(status_code=403, detail="Not subscribed to this podcast")
    return crud.list_episodes(db, podcast_id)

@app.post("/podcasts/{podcast_id}/fetch-latest")
def fetch_latest_episode(
    podcast_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Trigger a fetch/download/transcribe/summarize pass for the latest
    episode of the given podcast and return the newly created Episode.
    """
    # make sure the user is subscribed
    if not crud.is_user_subscribed(db, current_user.id, podcast_id):
        raise HTTPException(status_code=403, detail="Not subscribed")
    return crud.fetch_and_process_latest(db, podcast_id)