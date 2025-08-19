# ======================== rss_handler.py ========================
from __future__ import annotations

import feedparser
import requests
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict
from html.parser import HTMLParser

# --- HTTP & parsing guards ---
DEFAULT_TIMEOUT = 20
USER_AGENT = "PodcastSummarizerBot/1.0 (+https://metaldoglabs.ai)"
MAX_TRANSCRIPT_BYTES = 2 * 1024 * 1024   # 2 MB safety cap
MIN_TRANSCRIPT_WORDS = 100               # avoid tiny/junky snippets


# --- Simple HTML -> text stripper (no external deps) ---
class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._buf: List[str] = []
    def handle_data(self, d: str) -> None:
        self._buf.append(d)
    def get_text(self) -> str:
        return "".join(self._buf)

def _strip_html(html: str) -> str:
    if not html:
        return ""
    s = _HTMLStripper()
    try:
        s.feed(html)
        return " ".join(s.get_text().split()).strip()
    except Exception:
        # best-effort fallback
        return " ".join(html.split()).strip()


# ---------------- Core feed fetch/parse ----------------
def parse_feed(feed_url: str):
    """Fetch and parse RSS; returns the full FeedParserDict (not just entries)."""
    resp = requests.get(
        feed_url,
        headers={"User-Agent": USER_AGENT},
        timeout=DEFAULT_TIMEOUT
    )
    resp.raise_for_status()
    return feedparser.parse(resp.content)

def list_entries(feed_or_url) -> list:
    """Accept a feed dict or URL and return entries list."""
    if isinstance(feed_or_url, str):
        return parse_feed(feed_or_url).entries
    return getattr(feed_or_url, "entries", []) or []


# ---------------- Episode field extractors ----------------
def get_guid(entry) -> str:
    """Best-effort GUID for duplicate prevention."""
    return entry.get("id") or entry.get("guid") or entry.get("link")

def get_pub_date(entry) -> Optional[datetime]:
    """Parse published/updated date to tz-aware UTC datetime if available."""
    tp = getattr(entry, "published_parsed", None) or entry.get("published_parsed")
    if tp:
        return datetime(*tp[:6], tzinfo=timezone.utc)
    tp = getattr(entry, "updated_parsed", None) or entry.get("updated_parsed")
    if tp:
        return datetime(*tp[:6], tzinfo=timezone.utc)
    return None

def get_audio_url(entry) -> Optional[str]:
    """Prefer enclosure audio; fallback to first audio-like link or entry link."""
    encl = getattr(entry, "enclosures", None) or entry.get("enclosures")
    if encl:
        href = encl[0].get("href")
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
        s = str(itunes_duration).strip()
        if s.isdigit():
            return int(s)
        parts = [int(p) for p in s.split(":")]
        if len(parts) == 3:
            h, m, sec = parts
            return h * 3600 + m * 60 + sec
        if len(parts) == 2:
            m, sec = parts
            return m * 60 + sec
    except Exception:
        return None
    return None

def get_duration_seconds(entry) -> Optional[int]:
    return _parse_duration(getattr(entry, "itunes_duration", None) or entry.get("itunes_duration"))

def get_image_url(entry) -> Optional[str]:
    """Try common places episodes store images."""
    img_attr = getattr(entry, "image", None)
    if img_attr and isinstance(img_attr, dict) and img_attr.get("href"):
        return img_attr["href"]

    if entry.get("itunes_image"):
        img = entry.get("itunes_image")
        if isinstance(img, dict):
            return img.get("href") or img.get("url")
        return img

    thumbs = entry.get("media_thumbnail") or []
    if isinstance(thumbs, list) and thumbs and thumbs[0].get("url"):
        return thumbs[0]["url"]

    media = entry.get("media_content") or []
    if isinstance(media, list) and media and media[0].get("url"):
        return media[0]["url"]

    return None


# ---------------- Metadata (summary & transcript) extractors ----------------
def extract_summary_pair(entry) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (summary_plain, summary_html).
    We prefer entry.summary_detail.value as HTML when present.
    Plain text is always the stripped version (for LLM/UI).
    """
    html = None
    # Prefer explicit summary_detail.value (usually HTML)
    if entry.get("summary_detail") and isinstance(entry["summary_detail"], dict):
        html = (entry["summary_detail"].get("value") or "").strip()
    # Fallback to entry.summary if summary_detail missing
    if not html:
        html = (entry.get("summary") or "").strip()

    # Normalize
    html = html or None
    plain = _strip_html(html) if html else None
    return (plain if plain else None, html)


def _fetch_text_if_small(url: str) -> Optional[Tuple[str, str]]:
    """
    HEAD then GET a URL if smaller than MAX_TRANSCRIPT_BYTES.
    Returns (raw_text, content_type) or None if too large/error.
    """
    try:
        head = requests.head(url, allow_redirects=True, timeout=DEFAULT_TIMEOUT,
                             headers={"User-Agent": USER_AGENT})
        ctype = (head.headers.get("Content-Type") or "").lower()
        clen = head.headers.get("Content-Length")
        if clen and int(clen) > MAX_TRANSCRIPT_BYTES:
            return None

        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        return (r.text, ctype)
    except Exception:
        return None


def extract_transcript_pair(entry) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to discover an RSS-provided transcript.
    Returns (transcript_plain, transcript_html_raw) where:
      - transcript_plain is *plain text* (HTML stripped if needed),
      - transcript_html_raw is the raw body we fetched (if it was HTML), else None.

    We never write this into Episode.transcript (that is reserved for ASR).
    Store transcript_html_raw into Episode.transcript_html for user preview/choice.
    """
    # 1) Look for explicit transcript links (Podcasting 2.0 namespaces often use rel/type hints)
    for link in entry.get("links", []):
        rel = (link.get("rel") or "").lower()
        type_ = (link.get("type") or "").lower()
        href = link.get("href")
        if not href:
            continue
        if "transcript" in rel or "transcript" in type_:
            fetched = _fetch_text_if_small(href)
            if fetched:
                raw, ctype = fetched
                plain = _strip_html(raw) if ("html" in ctype or "xml" in ctype) else " ".join(raw.split()).strip()
                if len(plain.split()) >= MIN_TRANSCRIPT_WORDS:
                    # keep raw only if originally HTML-like; otherwise raw==plain text file
                    html_raw = raw if ("html" in ctype or "xml" in ctype) else None
                    return plain, html_raw

    # 2) Some feeds embed long-form transcript in content/summary; only accept if it's long enough
    # Use content first (richer than summary)
    if entry.get("content"):
        raw = (entry["content"][0].get("value") or "").strip()
        plain = _strip_html(raw)
        if len(plain.split()) >= MIN_TRANSCRIPT_WORDS:
            return plain, raw  # raw likely HTML

    # 3) As a last resort, if summary looks like a transcript and is long enough
    if entry.get("summary_detail"):
        raw = (entry["summary_detail"].get("value") or "").strip()
        plain = _strip_html(raw)
        if len(plain.split()) >= MIN_TRANSCRIPT_WORDS:
            return plain, raw

    return None, None


# ---------------- High-level payload for CRUD ingest ----------------
def build_episode_payload(entry) -> Dict[str, object]:
    """
    Build a dict ready for CRUD upsert.
    Do NOT include ASR transcript; keep RSS metadata separate:
      - summary: plain (LLM/UI ready)
      - summary_html: raw HTML (optional)
      - transcript_html: raw HTML transcript if found (optional)
    """
    guid = get_guid(entry)
    title = entry.get("title")
    pub_date = get_pub_date(entry)
    audio_url = get_audio_url(entry)
    duration_seconds = get_duration_seconds(entry)
    image_url = get_image_url(entry)

    summary_plain, summary_html = extract_summary_pair(entry)
    transcript_plain, transcript_html = extract_transcript_pair(entry)

    # We return both plain & html for summary, but only HTML for transcript
    # (plain transcript is useful to preview to the user, but should not be stored
    #  in Episode.transcript; the CRUD layer can surface it without persisting).
    payload: Dict[str, object] = {
        "guid": guid,
        "title": title,
        "pub_date": pub_date,
        "audio_url": audio_url,
        "duration_seconds": duration_seconds,
        "image_url": image_url,
        "summary": summary_plain,        # Episode.summary
        "summary_html": summary_html,    # Episode.summary_html
        "transcript_html": transcript_html,  # Episode.transcript_html (raw)
        # Add-ons for UI convenience (not for DB columns unless you want them):
        "has_rss_transcript": bool(transcript_html),
        "rss_transcript_preview": (transcript_plain[:2000] if transcript_plain else None),
    }
    return payload


def harvest_feed_metadata(feed_url: str, limit: Optional[int] = None) -> List[Dict[str, object]]:
    """
    Fetch feed and return a list of episode payloads (most recent first).
    'limit' trims the list for faster UI population.
    """
    feed = parse_feed(feed_url)
    entries = list_entries(feed)

    # Many feeds are already reverse-chron; keep order but be defensive:
    def _key(e):
        dt = get_pub_date(e)
        # newer first; None goes last
        return (dt is not None, dt)

    sorted_entries = sorted(entries, key=_key, reverse=True)
    if isinstance(limit, int) and limit > 0:
        sorted_entries = sorted_entries[:limit]

    return [build_episode_payload(e) for e in sorted_entries]
