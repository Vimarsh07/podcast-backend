# crud.py

import os
from sqlalchemy.orm import Session
from datetime import datetime
from fastapi import HTTPException, status

from model import User, Podcast, UserPodcast, Episode
from auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    decode_access_token,
)
import downloader
import rss_handler
import transcriber
import summarizer
import tempfile


def create_user(db: Session, email: str, password: str) -> User:
    """Registers a new user, hashing their password."""
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Email already registered")
    hashed = get_password_hash(password)
    user = User(email=email, password_hash=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> str | None:
    """Verifies credentials and returns a JWT, or None if invalid."""
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        return None
    token = create_access_token({"sub": str(user.id)})
    return token


def get_current_user(db: Session, token: str) -> User:
    """
    Decodes a JWT and returns the User instance.
    Raises 401 if token is invalid or user not found.
    """
    user_id = decode_access_token(token)
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")
    user = db.query(User).get(int(user_id))
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")
    return user


def get_user_podcasts(db: Session, user_id: int) -> list[Podcast]:
    """Returns a list of Podcast objects the user is subscribed to."""
    subs = (
        db.query(UserPodcast)
        .filter(UserPodcast.user_id == user_id)
        .all()
    )
    return [sub.podcast for sub in subs]


def subscribe_podcast(db: Session, user_id: int, feed_url: str) -> UserPodcast:
    """
    Subscribes a user to a podcast.
    Creates the Podcast record if it doesn't already exist.
    """
    # 1) fetch or create Podcast
    podcast = db.query(Podcast).filter(Podcast.feed_url == feed_url).first()
    if not podcast:
        podcast = Podcast(feed_url=feed_url, last_fetched=None)
        db.add(podcast)
        db.commit()
        db.refresh(podcast)

    # 2) fetch or create subscription
    sub = (
        db.query(UserPodcast)
        .filter(
            UserPodcast.user_id == user_id,
            UserPodcast.podcast_id == podcast.id,
        )
        .first()
    )
    if sub:
        return sub

    sub = UserPodcast(
        user_id=user_id,
        podcast_id=podcast.id,
        subscribed_at=datetime.utcnow(),
    )
    db.add(sub)
    db.commit()
    db.refresh(sub)
    return sub


def is_user_subscribed(db: Session, user_id: int, podcast_id: int) -> bool:
    """Returns True if the user has subscribed to the given podcast."""
    return (
        db.query(UserPodcast)
        .filter(
            UserPodcast.user_id == user_id,
            UserPodcast.podcast_id == podcast_id,
        )
        .count()
        > 0
    )


def list_episodes(db: Session, podcast_id: int) -> list[Episode]:
    """
    Returns all Episode objects for the given podcast, 
    ordered by publication date descending.
    """
    return (
        db.query(Episode)
        .filter(Episode.podcast_id == podcast_id)
        .order_by(Episode.pub_date.desc())
        .all()
    )

def fetch_and_process_latest(db: Session, podcast_id: int) -> Episode:
    """
    1. Load the Podcast by ID
    2. Parse its RSS feed and pick the newest entry
    3. Skip if we've already stored that GUID
    4. Download the audio, transcribe, summarize
    5. Persist a new Episode record and return it
    """
    # 1. Lookup podcast
    podcast = db.query(Podcast).get(podcast_id)
    if not podcast:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Podcast not found")

    # 2. Parse RSS entries
    entries = rss_handler.parse_feed(podcast.feed_url)
    if not entries:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No entries in feed")
    latest = entries[0]
    guid = latest.get("id") or latest.get("guid") or latest.get("link")
    
    # 3. Skip if already processed
    existing = (
        db.query(Episode)
          .filter_by(podcast_id=podcast_id, guid=guid)
          .first()
    )
    if existing:
        return existing

    # 4. Download audio to a temp file
    #    use enclosure link or default to entry.link
    audio_url = None
    for link in latest.get("links", []):
        if link.get("type", "").startswith("audio"):
            audio_url = link["href"]
            break
    if not audio_url:
        audio_url = latest.get("link")
    temp_dir = tempfile.gettempdir()
    filename = f"podcast_{podcast_id}_{guid}.mp3"
    output_path = os.path.join(temp_dir, filename)
    downloader.download_audio(audio_url, output_path)

    # 5. Transcribe (returns list of word dicts) 
    word_segments = transcriber.transcribe(output_path)
    # join into a single transcript string
    transcript_text = " ".join(w["text"] for w in word_segments)

    # 6. Summarize via OpenAI helper
    summary_text = summarizer.summarize_with_openai(transcript_text)

    # 7. Persist Episode
    ep = Episode(
        podcast_id=podcast_id,
        guid=guid,
        title=latest.get("title"),
        pub_date=latest.get("published_parsed") and datetime(*latest["published_parsed"][:6]),
        transcript=transcript_text,
        summary=summary_text,
        audio_url=audio_url
    )
    db.add(ep)
    db.commit()
    db.refresh(ep)

    # 8. Update podcast.last_fetched
    podcast.last_fetched = datetime.utcnow()
    db.commit()

    return ep