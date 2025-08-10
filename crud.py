# ======================== crud.py ========================
import os
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

import replicate

from db import SessionLocal
from model import User, Podcast, UserPodcast, Episode, TranscriptStatus
from auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    decode_access_token,
)

import rss_handler
from rss_handler import (
    extract_transcript_if_present,
    get_guid,
    get_pub_date,
    get_audio_url,
    get_duration_seconds,
    get_image_url,
)

import summarizer

# ─── Replicate Client ─────────────────────────────────────────────────────────
replicate_client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

# ─── Logger Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#                               USER MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

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

def authenticate_user(db: Session, email: str, password: str) -> Optional[str]:
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

# ─────────────────────────────────────────────────────────────────────────────
#                        SUBSCRIPTION / PODCASTS / EPISODES
# ─────────────────────────────────────────────────────────────────────────────

def get_user_podcasts(db: Session, user_id: int) -> List[Podcast]:
    """Returns a list of Podcast objects the user is subscribed to."""
    subs = db.query(UserPodcast).filter(UserPodcast.user_id == user_id).all()
    podcasts = [sub.podcast for sub in subs]
    logger.info(f"✅ Retrieved {len(podcasts)} subscriptions for user_id={user_id}")
    return podcasts

def subscribe_podcast(db: Session, user_id: int, feed_url: str, title: Optional[str] = None) -> UserPodcast:
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

    sub = db.query(UserPodcast).filter_by(user_id=user_id, podcast_id=podcast.id).first()
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
    count = db.query(UserPodcast).filter_by(user_id=user_id, podcast_id=podcast_id).count()
    subscribed = count > 0
    logger.info(f"ℹ️  is_user_subscribed user_id={user_id}, podcast_id={podcast_id}: {subscribed}")
    return subscribed

def unsubscribe_podcast(db: Session, user_id: int, podcast_id: int) -> None:
    """Removes a user’s subscription to a podcast."""
    db.query(UserPodcast).filter_by(user_id=user_id, podcast_id=podcast_id).delete()
    db.commit()
    logger.info(f"✅ Unsubscribed user_id={user_id} from podcast_id={podcast_id}")

def list_episodes(db: Session, podcast_id: int) -> List[Episode]:
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

# ─────────────────────────────────────────────────────────────────────────────
#                         METADATA INGEST (NO TRANSCRIPTION)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_latest_metadata_async(podcast_id: int, limit: int = 10):
    """
    Background-friendly wrapper to ingest the latest N episodes' metadata (no transcription).
    """
    logger.info(f"▶️ fetch_latest_metadata_async: podcast_id={podcast_id}, limit={limit}")
    with SessionLocal() as db:
        try:
            ingest_latest_metadata(db, podcast_id, limit=limit)
        except Exception as e:
            logger.error(f"❌ Async metadata ingest failed for podcast_id={podcast_id}: {e}")

def sync_all_podcasts_metadata_async(limit: int = 10):
    """
    Periodic refresher for ALL podcasts—keeps the episode lists up-to-date (metadata only).
    """
    logger.info("▶️ sync_all_podcasts_metadata_async: scanning all podcasts")
    with SessionLocal() as db:
        for p in db.query(Podcast).all():
            try:
                ingest_latest_metadata(db, p.id, limit=limit)
            except Exception as e:
                logger.error(f"❌ Metadata ingest for podcast {p.id} failed: {e}")

def ingest_latest_metadata(db: Session, podcast_id: int, limit: int = 10) -> List[Episode]:
    """
    Fetch RSS and UPSERT metadata for latest N episodes. No transcription here.
    Populates: guid, title, pub_date, summary (short), audio_url, + optional duration/image/status.
    """
    logger.info(f"▶️ ingest_latest_metadata: podcast_id={podcast_id}, limit={limit}")

    podcast = db.query(Podcast).get(podcast_id)
    if not podcast:
        logger.error(f"❌ Podcast not found id={podcast_id}")
        return []

    # Parse RSS
    try:
        feed = rss_handler.parse_feed(podcast.feed_url)
        entries = getattr(feed, "entries", feed)
        logger.info(f"✅ RSS parsed ({len(entries)} entries) for feed={podcast.feed_url}")
    except Exception as e:
        logger.error(f"❌ RSS parsing failed for feed={podcast.feed_url}: {e}")
        return []

    if not entries:
        logger.warning(f"⚠️ No entries in feed {podcast.feed_url}")
        return []

    processed: List[Episode] = []
    newest_dt: Optional[datetime] = None

    for entry in entries[:limit]:
        guid = get_guid(entry) or f"{podcast_id}:{entry.get('title','')}-{entry.get('published','')}"
        pub_dt = get_pub_date(entry)
        audio_url = get_audio_url(entry)
        duration_seconds = get_duration_seconds(entry)
        image_url = get_image_url(entry)

        # UPSERT by (podcast_id, guid)
        ep = db.query(Episode).filter_by(podcast_id=podcast_id, guid=guid).first()
        is_new = False
        if not ep:
            ep = Episode(podcast_id=podcast_id, guid=guid)
            db.add(ep)
            is_new = True

        ep.title = entry.get("title")
        ep.pub_date = pub_dt
        feed_summary = entry.get("summary") or entry.get("subtitle")

       # ✅ Only take the RSS summary if we don't already have an ASR-produced summary.
       #    Concretely: if there's no summary yet, or the episode has NOT been completed.
        if not ep.summary or getattr(ep, "transcript_status", None) in (None, TranscriptStatus.NOT_REQUESTED, TranscriptStatus.FAILED):
           ep.summary = feed_summary
        ep.audio_url = audio_url

        if hasattr(Episode, "duration_seconds"):
            ep.duration_seconds = duration_seconds
        if hasattr(Episode, "image_url"):
            ep.image_url = image_url

        # If feed embeds a transcript, store it if we don't already have one
        try:
            maybe_transcript = extract_transcript_if_present(entry)
        except Exception:
            maybe_transcript = None
        if maybe_transcript and not ep.transcript:
            ep.transcript = maybe_transcript

        # ✅ Only set NOT_REQUESTED when the episode is new or still in NOT_REQUESTED/None
        if hasattr(Episode, "transcript_status"):
            if is_new or getattr(ep, "transcript_status", None) in (None, TranscriptStatus.NOT_REQUESTED):
                if not ep.transcript:  # only if we don't have any transcript yet
                    ep.transcript_status = TranscriptStatus.NOT_REQUESTED
            # Do NOT touch status if it's QUEUED/TRANSCRIBING/COMPLETED/FAILED

        processed.append(ep)
        if pub_dt and (newest_dt is None or pub_dt > newest_dt):
            newest_dt = pub_dt

    if newest_dt:
        podcast.last_fetched = newest_dt

    db.commit()
    for ep in processed:
        db.refresh(ep)

    logger.info(f"✅ Ingested/updated {len(processed)} episodes (metadata only)")
    return processed

# ─────────────────────────────────────────────────────────────────────────────
#                     TRANSCRIBE + SUMMARIZE (ON-DEMAND)
# ─────────────────────────────────────────────────────────────────────────────

def _summarize_safely(text: str, max_words: int = 500) -> str:
    """Call your summarizer with max_words if available."""
    try:
        return summarizer.summarize_with_openai(text, max_words=max_words)
    except TypeError:
        return summarizer.summarize_with_openai(text)

def _get_audio_url_for(episode_id: int) -> str:
    with SessionLocal() as db:
        ep = db.query(Episode).get(episode_id)
        if not ep or not ep.audio_url:
            raise RuntimeError(f"Episode {episode_id} missing audio_url")
        return ep.audio_url

def transcribe_and_summarize_episode_async(episode_id: int, summary_words: int = 500):
    """
    Background-friendly wrapper executed by FastAPI BackgroundTasks.
    Uses short-lived DB sessions before/after the long compute.
    """
    # 1) Mark TRANSCRIBING quickly in its own transaction
    with SessionLocal() as db:
        ep = db.query(Episode).get(episode_id)
        if not ep:
            logger.error(f"❌ Episode not found id={episode_id}")
            return
        if hasattr(Episode, "transcript_status"):
            ep.transcript_status = TranscriptStatus.TRANSCRIBING
        db.commit()

    # 2) Long-running work WITHOUT any session held
    try:
        logger.info(f"▶️ Replicate: transcribing episode_id={episode_id}")
        words: List[dict] = replicate_client.run(
            # keep your pinned model/version here
            "vimarsh07/podcast-transcriber:190a68e5493e182db5dbd2730e0ec8607c9db5da31a5883d73f14fb7c73cfe82",
            input={"audio_url": _get_audio_url_for(episode_id)},
        )
        transcript_text = " ".join((w.get("text") or "").strip() for w in words).strip()
        logger.info("✅ Replicate transcription complete")
    except Exception as e:
        logger.error(f"❌ Replicate transcription error for episode_id={episode_id}: {e}")
        with SessionLocal() as db:
            ep = db.query(Episode).get(episode_id)
            if ep and hasattr(Episode, "transcript_status"):
                ep.transcript_status = TranscriptStatus.FAILED
                db.commit()
        return

    try:
        summary_text = _summarize_safely(transcript_text, max_words=summary_words)
        logger.info("✅ Summarization complete")
    except Exception as e:
        logger.error(f"❌ Summarization error for episode_id={episode_id}: {e}")
        summary_text = ""

    # 3) Save results in a fresh session; retry on transient DB failures
    import time
    for attempt in range(3):
        try:
            with SessionLocal() as db:
                ep = db.query(Episode).get(episode_id)
                if not ep:
                    return
                ep.transcript = transcript_text
                ep.summary = summary_text
                if hasattr(Episode, "transcript_status"):
                    ep.transcript_status = TranscriptStatus.COMPLETED
                db.commit()
            logger.info(f"✅ Episode updated id={episode_id} (transcript + summary)")
            break
        except Exception as e:
            logger.error(f"⚠️ DB write attempt {attempt+1} failed for episode_id={episode_id}: {e}")
            time.sleep(1.5 * (attempt + 1))
            if attempt == 2:
                with SessionLocal() as db2:
                    ep2 = db2.query(Episode).get(episode_id)
                    if ep2 and hasattr(Episode, "transcript_status"):
                        ep2.transcript_status = TranscriptStatus.FAILED
                        db2.commit()
                raise
