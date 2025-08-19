# ======================== crud.py ========================
import os
import logging
from datetime import datetime
from typing import List, Optional

from uuid import UUID
from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from db import SessionLocal
from model import (
    User,
    Podcast,
    UserPodcast,
    Episode,
    TranscriptStatus,
    TranscriptOrigin,   # <-- NEW
)
from auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    decode_access_token,
)

import summarizer

from rss_handler import harvest_feed_metadata, _strip_html

from deepgram_transcriber import transcribe_with_deepgram


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
    Works for UUID (current) and legacy int ids stored in the token.
    """
    user_sub = decode_access_token(token)
    if not user_sub:
        logger.error("❌ Invalid token")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")

    # Coerce subject to the right PK type
    pk = None
    s = str(user_sub)
    try:
        pk = UUID(s)            # preferred: users.id is UUID
    except ValueError:
        try:
            pk = int(s)         # legacy: if you ever had integer ids
        except ValueError:
            logger.error(f"❌ Bad token subject (not UUID or int): {s!r}")
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")

    # SQLAlchemy 1.4+ has Session.get; fall back to query().get for older code
    try:
        user = db.get(User, pk)   # type: ignore[attr-defined]
    except AttributeError:
        user = db.query(User).get(pk)

    if not user:
        logger.error(f"❌ User not found id={pk!r}")
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

    sub = UserPodcast(user_id=user_id, podcast_id=podcast.id)
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
    (main.py handles shaping/selection payload)
    """
    episodes = (
        db.query(Episode)
        .filter(Episode.podcast_id == podcast_id)
        .order_by(Episode.pub_date.desc().nullslast())
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
    podcast = db.query(Podcast).get(podcast_id)
    if not podcast:
        return []

    # structured payloads from rss_handler
    payloads = harvest_feed_metadata(podcast.feed_url, limit=limit)

    processed: List[Episode] = []
    newest_dt: Optional[datetime] = None

    for p in payloads:
        guid = p["guid"]
        ep = db.query(Episode).filter_by(podcast_id=podcast_id, guid=guid).first()
        is_new = False
        if not ep:
            ep = Episode(podcast_id=podcast_id, guid=guid)
            db.add(ep)
            is_new = True

        ep.title = p.get("title")
        ep.pub_date = p.get("pub_date")
        ep.audio_url = p.get("audio_url")
        ep.duration_seconds = p.get("duration_seconds")
        ep.image_url = p.get("image_url")

        # CHANGED: store cleaned metadata summary in summary_html (PLAIN TEXT).
        # Do NOT touch ep.summary (reserved for ASR summary).
        meta_plain = (p.get("summary") or "").strip()
        meta_raw   = (p.get("summary_html") or "").strip()
        cleaned = meta_plain or (_strip_html(meta_raw) if meta_raw else "")
        ep.summary_html = cleaned or None

        # Keep feed transcript (raw HTML) only in transcript_html
        rss_t_html = p.get("transcript_html")
        if rss_t_html:
            ep.transcript_html = rss_t_html

        # Initialize status for new rows only; don't clobber active states
        if is_new and ep.transcript_status in (None, TranscriptStatus.NOT_REQUESTED):
            ep.transcript_status = TranscriptStatus.NOT_REQUESTED

        processed.append(ep)
        dt = p.get("pub_date")
        if dt and (newest_dt is None or dt > newest_dt):
            newest_dt = dt

    if newest_dt:
        podcast.last_fetched = newest_dt

    db.commit()
    for ep in processed:
        db.refresh(ep)
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


def assemble_plain_transcript(words: List[dict]) -> str:
    """
    Convert Deepgram's normalized words into a readable transcript with speaker turns.
    Each word item: {start, end, text, speaker}
    """
    out: List[str] = []
    last_speaker = None
    line: List[str] = []

    for w in words:
        spk = w.get("speaker")
        txt = (w.get("text") or "").strip()
        if not txt:
            continue

        if spk != last_speaker:
            if line:
                out.append(" ".join(line).strip())
                line = []
            if spk is not None:
                out.append(f"\n[Speaker {spk}]")
            last_speaker = spk

        line.append(txt)

    if line:
        out.append(" ".join(line).strip())
    return "\n".join(out).strip()


def transcribe_and_summarize_episode_async(episode_id: int, summary_words: int = 500):
    """
    Background-friendly wrapper executed by FastAPI BackgroundTasks.
    Uses Deepgram for transcription + diarization, then your summarizer.
    """
    # 0) Fetch audio_url up front
    with SessionLocal() as db:
        ep = db.query(Episode).get(episode_id)
        if not ep:
            logger.error(f"❌ Episode not found id={episode_id}")
            return
        audio_url = getattr(ep, "audio_url", None)

    if not audio_url:
        logger.error(f"❌ Episode id={episode_id} has no audio_url")
        with SessionLocal() as db:
            ep = db.query(Episode).get(episode_id)
            if ep and hasattr(Episode, "transcript_status"):
                ep.transcript_status = TranscriptStatus.FAILED
                db.commit()
        return

    # 1) Mark TRANSCRIBING
    with SessionLocal() as db:
        ep = db.query(Episode).get(episode_id)
        if not ep:
            logger.error(f"❌ Episode not found id={episode_id}")
            return
        if hasattr(Episode, "transcript_status"):
            ep.transcript_status = TranscriptStatus.TRANSCRIBING
        db.commit()

    # 2) Do the long-running work (no session held)
    try:
        logger.info(f"▶️ Deepgram: transcribing episode_id={episode_id}")

        dg_result = transcribe_with_deepgram(
            audio_url=audio_url,
            language=os.getenv("DEEPGRAM_LANGUAGE", "en"),
            model=os.getenv("DEEPGRAM_MODEL", "nova-2"),
            diarize=True,
            smart_format=True,
            punctuate=True,
            paragraphs=True,
            utterances=True,
            num_speakers=None,
        )

        words: List[dict] = dg_result.get("words", [])
        if not words:
            raise RuntimeError("Deepgram returned no words")

        transcript_text = assemble_plain_transcript(words)
        logger.info("✅ Deepgram transcription complete")

    except Exception as e:
        logger.error(f"❌ Deepgram transcription error for episode_id={episode_id}: {e}")
        with SessionLocal() as db:
            ep = db.query(Episode).get(episode_id)
            if ep and hasattr(Episode, "transcript_status"):
                ep.transcript_status = TranscriptStatus.FAILED
                db.commit()
        return

    # 3) Summarize
    try:
        summary_text = _summarize_safely(transcript_text, max_words=summary_words)
        logger.info("✅ Summarization complete")
    except Exception as e:
        logger.error(f"❌ Summarization error for episode_id={episode_id}: {e}")
        summary_text = ""

    # 4) Persist results (with retries)
    import time
    for attempt in range(3):
        try:
            with SessionLocal() as db:
                ep = db.query(Episode).get(episode_id)
                if not ep:
                    return
                ep.transcript = transcript_text
                ep.summary = summary_text
                if hasattr(Episode, "transcript_origin"):
                    ep.transcript_origin = TranscriptOrigin.ASR   # <-- mark true ASR
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
