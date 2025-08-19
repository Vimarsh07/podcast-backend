# deepgram_transcriber.py
import os
import io
import json
import time
import logging
import tempfile
from typing import Any, Dict, Optional, Union, List
import requests

logger = logging.getLogger("deepgram_transcriber")
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"  # prerecorded

class DeepgramError(RuntimeError):
    pass

def _headers() -> Dict[str, str]:
    if not DEEPGRAM_API_KEY:
        raise DeepgramError("DEEPGRAM_API_KEY is not set")
    return {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Accept": "application/json",
    }

def _download_to_temp(audio_url: str) -> str:
    r = requests.get(audio_url, stream=True, timeout=(5, 60))
    r.raise_for_status()
    suffix = os.path.splitext(audio_url.split("?")[0])[1] or ".mp3"
    tf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    for chunk in r.iter_content(chunk_size=8192):
        tf.write(chunk)
    tf.flush(); tf.close()
    return tf.name

def _normalize_response(
    dg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Produce a structure compatible with your current pipeline:
      {
        device_asr: "deepgram",
        device_diar: "deepgram",
        num_items: int,
        estimated_speakers: int,
        speakers: [spk1, spk2, ...],
        words: [{start, end, text, speaker}]
      }
    Deepgram returns both 'words' and 'paragraphs/utterances' (with speaker).
    Strategy:
    - Use words for token-level timing.
    - Assign speaker from the best-matching utterance/paragraph segment.
    """
    # Pull words (start/end/text)
    words: List[Dict[str, Any]] = []
    speakers_set = set()

    # Prefer paragraphs. If missing, fall back to utterances. If missing, assign speaker=None.
    paragraphs = (dg.get("results", {}) or {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("paragraphs")
    utterances = (dg.get("results", {}) or {}).get("utterances")
    alts = (dg.get("results", {}) or {}).get("channels", [{}])[0].get("alternatives", [{}])

    # Extract a flat words list
    alt0 = alts[0] if alts else {}
    raw_words = alt0.get("words", []) or []
    # Build a lookup from time -> speaker using utterances/paragraphs
    speaker_spans: List[Dict[str, Any]] = []

    if paragraphs and isinstance(paragraphs, dict):
        for para in paragraphs.get("paragraphs", []):
            spk = para.get("speaker")
            start = float(para.get("start", 0.0))
            end = float(para.get("end", start))
            speaker_spans.append({"start": start, "end": end, "speaker": spk})
            if spk is not None:
                speakers_set.add(str(spk))

    if not speaker_spans and utterances and isinstance(utterances, list):
        for utt in utterances:
            spk = utt.get("speaker")
            start = float(utt.get("start", 0.0))
            end = float(utt.get("end", start))
            speaker_spans.append({"start": start, "end": end, "speaker": spk})
            if spk is not None:
                speakers_set.add(str(spk))

    def _speaker_for_time(t: float) -> Optional[str]:
        # find span that contains t; else nearest
        if not speaker_spans:
            return None
        for s in speaker_spans:
            if s["start"] <= t < s["end"]:
                return str(s["speaker"]) if s["speaker"] is not None else None
        # nearest by midpoint
        nearest = min(speaker_spans, key=lambda s: abs(((s["start"] + s["end"]) / 2.0) - t))
        return str(nearest["speaker"]) if nearest["speaker"] is not None else None

    for w in raw_words:
        start = float(w.get("start", 0.0))
        end = float(w.get("end", start))
        text = w.get("word", "")  # sometimes "punctuated_word" is present; using "word" is safer
        spk = _speaker_for_time((start + end) / 2.0)
        if spk:
            speakers_set.add(spk)
        words.append({"start": start, "end": end, "text": text, "speaker": spk})

    result = {
        "device_asr": "deepgram",
        "device_diar": "deepgram",
        "num_items": len(words),
        "estimated_speakers": len(speakers_set),
        "speakers": sorted(speakers_set),
        "words": words,
    }
    return result

def transcribe_with_deepgram(
    *,
    audio_url: Optional[str] = None,
    audio_file_path: Optional[str] = None,
    language: str = "en",
    model: str = "nova-2",             # Deepgram's high‑accuracy model
    diarize: bool = True,
    smart_format: bool = True,
    punctuate: bool = True,
    paragraphs: bool = True,
    utterances: bool = True,
    # Optional hints
    num_speakers: Optional[int] = None,
    retries: int = 2,
    timeout: int = 600,
) -> Dict[str, Any]:
    """
    One-shot call to Deepgram for ASR + diarization.
    - Supports remote URL or local path.
    - Returns normalized structure compatible with your previous output.
    """
    if not audio_url and not audio_file_path:
        raise DeepgramError("Provide either audio_url or audio_file_path")

    # Build query params
    params = {
        "model": model,
        "language": language,
        "diarize": "true" if diarize else "false",
        "smart_format": "true" if smart_format else "false",
        "punctuate": "true" if punctuate else "false",
        "paragraphs": "true" if paragraphs else "false",
        "utterances": "true" if utterances else "false",
    }
    if num_speakers is not None:
        # Deepgram will use this as a hint; if unsupported it will be ignored.
        params["diarize_speakers"] = str(int(num_speakers))

    # Send either JSON (for URL) or multipart (for raw bytes)
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 2):
        try:
            if audio_url:
                resp = requests.post(
                    DEEPGRAM_URL,
                    headers={**_headers(), "Content-Type": "application/json"},
                    params=params,
                    data=json.dumps({"url": audio_url}),
                    timeout=timeout,
                )
            else:
                # upload file
                with open(audio_file_path, "rb") as f:
                    files = {"audio": f}
                    resp = requests.post(
                        DEEPGRAM_URL,
                        headers=_headers(),
                        params=params,
                        files=files,
                        timeout=timeout,
                    )

            if resp.status_code >= 400:
                raise DeepgramError(f"Deepgram error {resp.status_code}: {resp.text[:500]}")

            dg = resp.json()
            return _normalize_response(dg)
        except Exception as e:
            last_err = e
            logger.warning(f"Deepgram attempt {attempt} failed: {e}")
            if attempt <= retries:
                time.sleep(min(2 ** attempt, 8))
            else:
                break
    raise DeepgramError(f"Failed Deepgram transcription after retries: {last_err}")
