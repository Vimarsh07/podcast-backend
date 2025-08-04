# crud.py

import os
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from db import SessionLocal
from fastapi import HTTPException, status
import replicate

from model import User, Podcast, UserPodcast, Episode
from auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    decode_access_token,
)
import rss_handler

import summarizer


replicate_client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])




# ─── Logger Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── User Management ─────────────────────────────────────────────────────────

def create_user(db: Session, email: str, password: str) -> User:
    """Registers a new user, hashing their password."""
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Email already registered")

    hashed = get_password_hash(password)
    user = User(email=email, password_hash=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info(f"✅ Created user id={user.id}, email={email}")
    return user

def authenticate_user(db: Session, email: str, password: str) -> str | None:
    """Verifies credentials and returns a JWT, or None if invalid."""
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        logger.warning(f"⚠️  Authentication failed for email={email}")
        return None

    token = create_access_token({"sub": str(user.id)})
    logger.info(f"✅ Authenticated user id={user.id}")
    return token

def get_current_user(db: Session, token: str) -> User:
    """
    Decodes a JWT and returns the User instance.
    Raises 401 if token is invalid or user not found.
    """
    user_id = decode_access_token(token)
    if not user_id:
        logger.error("❌ Invalid token")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")

    user = db.query(User).get(int(user_id))
    if not user:
        logger.error(f"❌ User not found id={user_id}")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found")

    logger.info(f"✅ get_current_user: id={user.id}, email={user.email}")
    return user

# ─── Subscription Management ─────────────────────────────────────────────────

def get_user_podcasts(db: Session, user_id: int) -> list[Podcast]:
    """Returns a list of Podcast objects the user is subscribed to."""
    subs = (
        db.query(UserPodcast)
        .filter(UserPodcast.user_id == user_id)
        .all()
    )
    podcasts = [sub.podcast for sub in subs]
    logger.info(f"✅ Retrieved {len(podcasts)} subscriptions for user_id={user_id}")
    return podcasts

def subscribe_podcast(db: Session, user_id: int, feed_url: str, title: str | None = None) -> UserPodcast:
    """
    Subscribes a user to a podcast.
    Creates the Podcast record if it doesn't already exist.
    """
    podcast = db.query(Podcast).filter(Podcast.feed_url == feed_url).first()
    if not podcast:
        podcast = Podcast(feed_url=feed_url, title=title, last_fetched=None)
        db.add(podcast)
        db.commit()
        db.refresh(podcast)
        logger.info(f"✅ Created Podcast id={podcast.id}, feed_url={feed_url}")

    sub = (
        db.query(UserPodcast)
        .filter_by(user_id=user_id, podcast_id=podcast.id)
        .first()
    )
    if sub:
        logger.info(f"ℹ️  User {user_id} already subscribed to podcast {podcast.id}")
        return sub

    sub = UserPodcast(user_id=user_id, podcast_id=podcast.id, subscribed_at=datetime.utcnow())
    db.add(sub)
    db.commit()
    db.refresh(sub)
    logger.info(f"✅ Subscribed user_id={user_id} to podcast_id={podcast.id}")
    return sub

def is_user_subscribed(db: Session, user_id: int, podcast_id: int) -> bool:
    """Returns True if the user has subscribed to the given podcast."""
    count = (
        db.query(UserPodcast)
        .filter_by(user_id=user_id, podcast_id=podcast_id)
        .count()
    )
    subscribed = count > 0
    logger.info(f"ℹ️  is_user_subscribed user_id={user_id}, podcast_id={podcast_id}: {subscribed}")
    return subscribed

def unsubscribe_podcast(db: Session, user_id: int, podcast_id: int) -> None:
    """Removes a user’s subscription to a podcast."""
    db.query(UserPodcast).filter_by(user_id=user_id, podcast_id=podcast_id).delete()
    db.commit()
    logger.info(f"✅ Unsubscribed user_id={user_id} from podcast_id={podcast_id}")

def list_episodes(db: Session, podcast_id: int) -> list[Episode]:
    """
    Returns all Episode objects for the given podcast,
    ordered by publication date descending.
    """
    episodes = (
        db.query(Episode)
        .filter(Episode.podcast_id == podcast_id)
        .order_by(Episode.pub_date.desc())
        .all()
    )
    logger.info(f"✅ Retrieved {len(episodes)} episodes for podcast_id={podcast_id}")
    return episodes

# ─── Background & Polling ────────────────────────────────────────────────────

def fetch_and_process_latest_async(podcast_id: int):
    logger.info(f"▶️ fetch_and_process_latest_async: podcast_id={podcast_id}")
    db = SessionLocal()
    try:
        _fetch_and_process_latest(db, podcast_id, only_if_new=False)
    except Exception as e:
        logger.error(f"❌ Async fetch_and_process_latest failed for podcast_id={podcast_id}: {e}")
    finally:
        db.close()

def sync_all_podcasts_async():
    logger.info("▶️ sync_all_podcasts_async: scanning all podcasts")
    db = SessionLocal()
    try:
        for p in db.query(Podcast).all():
            try:
                _fetch_and_process_latest(db, p.id, only_if_new=True)
            except Exception as e:
                logger.error(f"❌ Polling podcast {p.id} failed: {e}")
    finally:
        db.close()

def _fetch_and_process_latest(
    db: Session,
    podcast_id: int,
    only_if_new: bool = False
) -> Episode | None:
    logger.info(f"▶️ _fetch_and_process_latest: podcast_id={podcast_id}, only_if_new={only_if_new}")

    # 1) Load podcast record
    podcast = db.query(Podcast).get(podcast_id)
    if not podcast:
        logger.error(f"❌ Podcast not found id={podcast_id}")
        return None

    # 2) Fetch & parse RSS
    try:
        entries = rss_handler.parse_feed(podcast.feed_url)
        logger.info(f"✅ RSS parsed ({len(entries)} entries) for feed={podcast.feed_url}")
    except Exception as e:
        logger.error(f"❌ RSS parsing failed for feed={podcast.feed_url}: {e}")
        return None
    if not entries:
        logger.warning(f"⚠️ No entries in feed {podcast.feed_url}")
        return None

    latest = entries[0]
    guid = latest.get("id") or latest.get("guid") or latest.get("link")
    pub_dt = datetime(*latest.published_parsed[:6])

    # 3) Skip if already fetched
    if only_if_new and podcast.last_fetched and podcast.last_fetched >= pub_dt:
        logger.info(f"ℹ️ No new episode (last_fetched={podcast.last_fetched})")
        return None
 # 4) Determine audio URL from enclosure (fallback to links)
    if getattr(latest, "enclosures", None):
        audio_url = latest.enclosures[0]["href"]
    else:
        audio_url = next(
            (l["href"] for l in latest.get("links", [])
             if l.get("type", "").startswith("audio")),
            latest.get("link")
        )
    logger.info(f"ℹ️ Processing episode GUID={guid}, audio_url={audio_url}")
    # 5) Transcription via Replicate
    try:
        words: list[dict] = replicate_client.run(
            "vimarsh07/podcast-transcriber:190a68e5493e182db5dbd2730e0ec8607c9db5da31a5883d73f14fb7c73cfe82",  # ← replace with your model
            input={"audio_url": audio_url}
        )
        transcript = " ".join(w["text"] for w in words)
        logger.info("✅ Replicate transcription complete")
    except Exception as e:
        logger.error(f"❌ Replicate transcription error for GUID={guid}: {e}")
        return None

    # 6) Summarization
    try:
        summary = summarizer.summarize_with_openai(transcript)
        logger.info("✅ Summarization complete")
    except Exception as e:
        logger.error(f"❌ Summarization error for GUID={guid}: {e}")
        summary = ""

    # 7) Persist the new Episode
    try:
        ep = Episode(
            podcast_id=podcast_id,
            guid=guid,
            title=latest.get("title"),
            pub_date=pub_dt,
            transcript=transcript,
            summary=summary,
            audio_url=audio_url
        )
        db.add(ep)
        podcast.last_fetched = pub_dt
        db.commit()
        db.refresh(ep)
        logger.info(f"✅ Episode saved id={ep.id}")
        return ep
    except Exception as e:
        db.rollback()
        logger.error(f"❌ DB save error for GUID={guid}: {e}")
        return None