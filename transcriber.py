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

# ─── Load Speaker-Diarization Pipeline ───────────────────────────────────────
HF_TOKEN = os.getenv("HF_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=HF_TOKEN
    )
    logger.info("✅ Loaded pyannote speaker-diarization pipeline")
except Exception as e:
    logger.warning(f"⚠️ Could not load diarization pipeline: {e}")
    diarization_pipeline = None

# ─── Load Whisper Model ──────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"
try:
    whisper_model = WhisperModel(
        "base",
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )
    logger.info(f"✅ Loaded WhisperModel(base) on {DEVICE} with {COMPUTE_TYPE}")
except Exception as e:
    logger.error(f"❌ Failed to load WhisperModel: {e}")
    raise

def transcribe(audio_url: str) -> list[dict]:
    """
    Downloads the audio at `audio_url`, runs Whisper ASR + pyannote diarization,
    and returns a list of {start, end, text, speaker}.
    """
    logger.info(f"▶️ transcribe(): start for URL: {audio_url}")

    # 1) Download to temp file
    try:
        resp = requests.get(audio_url, stream=True, timeout=30)
        resp.raise_for_status()
        suffix = os.path.splitext(audio_url)[1] or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(resp.content)
            temp_path = tf.name
        logger.info(f"✅ Audio fetched to {temp_path}")
    except Exception as e:
        logger.error(f"❌ Failed to fetch audio: {e}")
        raise

    # 2) Whisper ASR
    try:
        segments, _ = whisper_model.transcribe(
            temp_path,
            beam_size=5,
            word_timestamps=True
        )
        words = [
            {"start": w.start, "end": w.end, "text": w.word}
            for seg in segments for w in seg.words
        ]
        logger.info(f"✅ ASR complete: {len(words)} words")
    except Exception as e:
        logger.error(f"❌ ASR failed: {e}")
        words = []

    # 3) Speaker diarization
    labeled = []
    if diarization_pipeline and words:
        try:
            logger.info("▶️ Running speaker diarization")
            diarization = diarization_pipeline({"audio": temp_path})
            for w in words:
                spk = next(
                    (
                        label
                        for turn, _, label in diarization.itertracks(yield_label=True)
                        if turn.start <= w["start"] < turn.end
                    ),
                    None
                )
                labeled.append({**w, "speaker": spk})
            logger.info(f"✅ Diarization complete: {len(labeled)} words labeled")
        except Exception as e:
            logger.error(f"❌ Diarization failed: {e}")
            labeled = [{**w, "speaker": None} for w in words]
    else:
        if not diarization_pipeline:
            logger.warning("⚠️ No diarization pipeline available")
        labeled = [{**w, "speaker": None} for w in words]

    # 4) Cleanup
    try:
        os.remove(temp_path)
        logger.info(f"🗑️ Deleted temp file: {temp_path}")
    except Exception:
        logger.warning("⚠️ Could not delete temp file")

    return labeled
