import os
import logging
import requests
import tempfile
import torch

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ─── Initialize Speaker‐Diarization Pipeline ─────────────────────────────────
HF_TOKEN = os.getenv("HF_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
try:
    diari_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=HF_TOKEN
    )
    logger.info("✅ Pyannote speaker-diarization pipeline loaded")
except Exception as e:
    logger.error(f"❌ Failed to load speaker-diarization pipeline: {e}")
    diari_pipeline = None

# ─── Whisper Model Cache ──────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"
_model: WhisperModel | None = None

def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        logger.info(f"Loading WhisperModel(base) on {DEVICE} with {COMPUTE_TYPE}")
        _model = WhisperModel("base", device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model

# ─── New helper: retrying download ─────────────────────────────────────────────
def _download_audio(audio_url: str, retries: int = 3) -> str:
    """
    Download audio_url to a temp file, retrying on failure.
    Returns the local file path.
    """
    for attempt in range(1, retries + 1):
        try:
            # connect timeout 5s, read timeout 60s
            resp = requests.get(audio_url, stream=True, timeout=(5, 60))
            resp.raise_for_status()
            suffix = os.path.splitext(audio_url)[1] or ".mp3"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                for chunk in resp.iter_content(chunk_size=8192):
                    tf.write(chunk)
                temp_path = tf.name
            logger.info(f"✅ Downloaded audio (attempt {attempt}) to {temp_path}")
            return temp_path
        except Exception as e:
            logger.warning(f"⚠️ Download attempt {attempt} failed: {e}")
    raise RuntimeError(f"Failed to download audio after {retries} attempts")

def transcribe(audio_url: str) -> list[dict]:
    logger.info(f"▶️ transcribe(): start for URL: {audio_url}")

    # ─── Download into temp file ────────────────────────────────────────────────
    try:
        temp_path = _download_audio(audio_url)
    except Exception as e:
        logger.error(f"❌ Audio download ultimately failed: {e}")
        raise

    # ─── Whisper ASR ──────────────────────────────────────────────────────────
    try:
        model = _get_model()
        segments, _ = model.transcribe(
            temp_path, beam_size=5, word_timestamps=True
        )
        words = [
            {"start": w.start, "end": w.end, "text": w.word}
            for seg in segments for w in seg.words
        ]
        logger.info(f"✅ Whisper ASR complete: {len(words)} words")
    except Exception as e:
        logger.error(f"❌ Whisper ASR failed: {e}")
        words = []

    # ─── Speaker diarization ─────────────────────────────────────────────────
    labeled = []
    if diari_pipeline and words:
        try:
            logger.info("▶️ Running speaker diarization...")
            diarization = diari_pipeline({"audio": temp_path})
            for w in words:
                spk = next(
                    (label for turn, _, label in diarization.itertracks(yield_label=True)
                     if turn.start <= w["start"] < turn.end),
                    None
                )
                labeled.append({**w, "speaker": spk})
            logger.info(f"✅ Speaker diarization complete: {len(labeled)} words labeled")
        except Exception as e:
            logger.error(f"❌ Diarization failed: {e}")
            labeled = [{**w, "speaker": None} for w in words]
    else:
        if not diari_pipeline:
            logger.warning("⚠️ No diarization pipeline available")
        labeled = [{**w, "speaker": None} for w in words]

    # ─── Cleanup ────────────────────────────────────────────────────────────────
    try:
        os.remove(temp_path)
        logger.info(f"🗑️ Temp file deleted: {temp_path}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to delete temp file: {e}")

    return labeled
