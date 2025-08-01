# ======================== rss_handler.py ========================
import feedparser
import requests


def parse_feed(feed_url: str):
    """Fetch and parse RSS feed entries"""
    return feedparser.parse(feed_url).entries


def extract_transcript_if_present(entry) -> str:
    """Return a transcript text if found in summary or a transcript link, else None"""
    # direct transcript link
    for link in entry.get("links", []):
        if 'transcript' in link.get('rel', ''):
            try:
                return requests.get(link['href']).text
            except:
                pass
    # embedded in summary_detail
    if entry.get("summary_detail"):
        text = entry["summary_detail"].get("value", "")
        if len(text.split()) > 100:
            return text
    # embedded in content
    if entry.get("content"):
        val = entry["content"][0].get("value", "")
        if len(val.split()) > 100:
            return val
    return None
