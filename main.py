from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from db import SessionLocal, init_db
from models import User, Podcast, UserPodcast, Episode
import crud   # you’ll create simple CRUD wrappers

# initialize tables
init_db()

app = FastAPI(title="Podcast Summarizer API")

# allow calls from your React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get a DB session per request
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/signup")
def signup(email: str, password: str, db: Session = Depends(get_db)):
    # hash password, create user...
    return crud.create_user(db, email, password)

@app.post("/login")
def login(email: str, password: str, db: Session = Depends(get_db)):
    # verify, return JWT token...
    token = crud.authenticate_user(db, email, password)
    if not token:
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": token, "token_type": "bearer"}

@app.get("/podcasts")
def list_podcasts(db: Session = Depends(get_db), token: str = Depends(crud.get_current_user)):
    user = crud.get_current_user(db, token)
    return crud.get_user_podcasts(db, user.id)

@app.post("/podcasts")
def subscribe_podcast(feed_url: str, db: Session = Depends(get_db), token: str = Depends(crud.get_current_user)):
    user = crud.get_current_user(db, token)
    return crud.subscribe_podcast(db, user.id, feed_url)

@app.get("/episodes/{podcast_id}")
def get_episodes(podcast_id: int, db: Session = Depends(get_db), token: str = Depends(crud.get_current_user)):
    user = crud.get_current_user(db, token)
    # verify subscription...
    return crud.list_episodes(db, podcast_id)
