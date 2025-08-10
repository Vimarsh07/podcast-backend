# ======================== rss_handler.py ========================
from __future__ import annotations
import feedparser
import requests
from datetime import datetime, timezone
from typing import Optional
from html.parser import HTMLParser

# --- HTTP & parsing guards ---
DEFAULT_TIMEOUT = 20
USER_AGENT = "PodcastSummarizerBot/1.0 (+https://metaldoglabs.ai)"
MAX_TRANSCRIPT_BYTES = 2 * 1024 * 1024   # 2 MB safety cap
MIN_TRANSCRIPT_WORDS = 100               # avoid junky snippets

# --- Simple HTML -> text stripper (no external deps) ---
class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._buf = []
    def handle_data(self, d): self._buf.append(d)
    def get_text(self): return "".join(self._buf)

def _strip_html(html: str) -> str:
    s = _HTMLStripper()
    try:
        s.feed(html)
        return s.get_text()
    except Exception:
        return html

# ---------------- Core feed helpers ----------------

def parse_feed(feed_url: str):
    """Fetch and parse RSS; returns the full FeedParserDict (not just entries)."""
    resp = requests.get(
        feed_url,
        headers={"User-Agent": USER_AGENT},
        timeout=DEFAULT_TIMEOUT
    )
    resp.raise_for_status()
    return feedparser.parse(resp.content)

def list_entries(feed_or_url):
    """Accept a feed dict or URL and return entries list."""
    if isinstance(feed_or_url, str):
        return parse_feed(feed_or_url).entries
    return getattr(feed_or_url, "entries", []) or []

def extract_transcript_if_present(entry) -> Optional[str]:
    """
    Return a *clean text* transcript if found:
    - Prefer explicit transcript links (by rel/type)
    - Otherwise fall back to embedded summary/content
    Applies HTML stripping and size/type/timeouts to be safe.
    """
    # 1) Explicit transcript links
    for link in entry.get("links", []):
        rel = (link.get("rel") or "").lower()
        type_ = (link.get("type") or "").lower()
        href = link.get("href")
        if not href:
            continue
        if "transcript" in rel or "transcript" in type_:
            try:
                head = requests.head(href, allow_redirects=True, timeout=DEFAULT_TIMEOUT,
                                     headers={"User-Agent": USER_AGENT})
                ctype = (head.headers.get("Content-Type") or "").lower()
                clen  = head.headers.get("Content-Length")
                if clen and int(clen) > MAX_TRANSCRIPT_BYTES:
                    continue  # too large; skip fetching
                # fetch body
                r = requests.get(href, headers={"User-Agent": USER_AGENT}, timeout=DEFAULT_TIMEOUT)
                r.raise_for_status()
                raw = r.text
                # Convert HTML to plain text if needed
                text = _strip_html(raw) if "html" in ctype else raw
                text = text.strip()
                if len(text.split()) >= MIN_TRANSCRIPT_WORDS:
                    return text
            except Exception:
                pass  # non-fatal; keep hunting

    # 2) Embedded summary_detail
    if entry.get("summary_detail"):
        raw = (entry["summary_detail"].get("value") or "").strip()
        txt = _strip_html(raw).strip()
        if len(txt.split()) >= MIN_TRANSCRIPT_WORDS:
            return txt

    # 3) Embedded content
    if entry.get("content"):
        raw = (entry["content"][0].get("value") or "").strip()
        txt = _strip_html(raw).strip()
        if len(txt.split()) >= MIN_TRANSCRIPT_WORDS:
            return txt

    return None

# ---------- Extra metadata helpers used by CRUD ingest ----------

def get_guid(entry) -> str:
    """Best-effort GUID for duplicate prevention."""
    return entry.get("id") or entry.get("guid") or entry.get("link")

def get_pub_date(entry) -> Optional[datetime]:
    """Parse published/updated date to tz-aware UTC datetime if available."""
    # feedparser may expose both attribute-style and dict-style
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
        if len(parts) == 3: h, m, sec = parts; return h * 3600 + m * 60 + sec
        if len(parts) == 2: m, sec = parts;   return m * 60 + sec
    except Exception:
        return None
    return None

def get_duration_seconds(entry) -> Optional[int]:
    return _parse_duration(getattr(entry, "itunes_duration", None) or entry.get("itunes_duration"))

def get_image_url(entry) -> Optional[str]:
    """Try common places episodes store images."""
    img_attr = getattr(entry, "image", None)
    if img_attr and img_attr.get("href"):
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
