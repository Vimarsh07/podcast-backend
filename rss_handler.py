# ======================== rss_handler.py ========================
from __future__ import annotations
import feedparser
import requests
from datetime import datetime, timezone
from typing import Optional

DEFAULT_TIMEOUT = 15
USER_AGENT = "PodcastSummarizerBot/1.0 (+https://yourdomain.example)"

def parse_feed(feed_url: str):
    """
    Fetch and parse RSS. Returns the FULL feed (FeedParserDict), not just entries,
    so callers can access feed-level metadata if needed.
    Backward-compatible helpers below still return .entries.
    """
    # Prefetch with requests so we can set headers/timeouts (some feeds block default UA).
    resp = requests.get(feed_url, headers={"User-Agent": USER_AGENT}, timeout=DEFAULT_TIMEOUT)
    resp.raise_for_status()
    return feedparser.parse(resp.content)

def list_entries(feed_or_url):
    """
    Convenience: accept either a feed dict (from parse_feed) or a URL and return entries list.
    """
    if isinstance(feed_or_url, str):
        return parse_feed(feed_or_url).entries
    return getattr(feed_or_url, "entries", []) or []

def extract_transcript_if_present(entry) -> Optional[str]:
    """Return a transcript text if found in summary/content or via a transcript link, else None."""
    # direct transcript link (rel may vary by publisher; check 'transcript' in rel OR type)
    for link in entry.get("links", []):
        rel = (link.get("rel") or "").lower()
        type_ = (link.get("type") or "").lower()
        if "transcript" in rel or "transcript" in type_:
            try:
                r = requests.get(link["href"], headers={"User-Agent": USER_AGENT}, timeout=DEFAULT_TIMEOUT)
                r.raise_for_status()
                text = r.text.strip()
                if len(text.split()) > 50:  # slight guard against empty files
                    return text
            except Exception:
                pass

    # embedded in summary_detail
    if entry.get("summary_detail"):
        text = (entry["summary_detail"].get("value") or "").strip()
        if len(text.split()) > 100:
            return text

    # embedded in content
    if entry.get("content"):
        val = (entry["content"][0].get("value") or "").strip()
        if len(val.split()) > 100:
            return val
    return None

# ---------- New helpers for your metadata-only ingest ----------

def get_guid(entry) -> str:
    """Best-effort GUID for duplicate prevention."""
    return entry.get("id") or entry.get("guid") or entry.get("link")

def get_pub_date(entry) -> Optional[datetime]:
    """Parse published date to tz-aware UTC datetime if available."""
    if getattr(entry, "published_parsed", None):
        # published_parsed is a time.struct_time
        return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    if getattr(entry, "updated_parsed", None):
        return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
    return None

def get_audio_url(entry) -> Optional[str]:
    """Prefer enclosure audio; fallback to first audio-like link or entry link."""
    if getattr(entry, "enclosures", None):
        href = entry.enclosures[0].get("href")
        if href:
            return href
    for l in entry.get("links", []):
        type_ = (l.get("type") or "").lower()
        if type_.startswith("audio") or "audio" in type_:
            return l.get("href")
    return entry.get("link")

def _parse_duration(itunes_duration: Optional[str]) -> Optional[int]:
    """Parse itunes:duration 'HH:MM:SS' / 'MM:SS' / 'SS' to seconds."""
    if not itunes_duration:
        return None
    try:
        s = itunes_duration.strip()
        if s.isdigit():
            return int(s)
        parts = [int(p) for p in s.split(":")]
        if len(parts) == 3:
            h, m, s = parts
            return h * 3600 + m * 60 + s
        if len(parts) == 2:
            m, s = parts
            return m * 60 + s
    except Exception:
        return None
    return None

def get_duration_seconds(entry) -> Optional[int]:
    return _parse_duration(getattr(entry, "itunes_duration", None) or entry.get("itunes_duration"))

def get_image_url(entry) -> Optional[str]:
    """Try common places episodes store images."""
    # feedparser often exposes any <image> or itunes:image in one of these:
    if getattr(entry, "image", None) and entry.image.get("href"):
        return entry.image["href"]
    if entry.get("itunes_image"):
        # some feeds store a URL or a dict; handle both
        img = entry.get("itunes_image")
        if isinstance(img, dict):
            return img.get("href") or img.get("url")
        return img
    # sometimes 'media_thumbnail' or 'media_content' exists
    thumbs = entry.get("media_thumbnail") or []
    if thumbs and isinstance(thumbs, list) and thumbs[0].get("url"):
        return thumbs[0]["url"]
    media = entry.get("media_content") or []
    if media and isinstance(media, list) and media[0].get("url"):
        return media[0]["url"]
    return None
