from huggingface_hub import HfApi, HfFolder
import os, logging, tempfile, requests, torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HF_HUB_TOKEN")
    or os.getenv("HUGGINGFACE_TOKEN")
)

# Optional: surface a helpful log if token is missing
if not HF_TOKEN:
    logger.warning("⚠️ No HF token found in env (HF_TOKEN/HF_HUB_TOKEN/HUGGINGFACE_TOKEN)")

# Try to persist token for huggingface_hub
try:
    if HF_TOKEN:
        HfFolder.save_token(HF_TOKEN)
        logger.info("✅ Hugging Face token saved to cache")
except Exception as e:
    logger.warning(f"⚠️ Could not persist HF token: {e}")

# Force a whoami check so failures are obvious
try:
    if HF_TOKEN:
        who = HfApi().whoami(token=HF_TOKEN)
        logger.info(f"👤 HF auth OK: {who.get('name') or who.get('email')}")
except Exception as e:
    logger.error(f"❌ HF auth failed (bad/missing token or network): {e}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"
_model: WhisperModel | None = None

def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        logger.info(f"Loading WhisperModel(base) on {DEVICE} with {COMPUTE_TYPE}")
        _model = WhisperModel("base", device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model

def _download_audio(audio_url: str, retries: int = 3) -> str:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(audio_url, stream=True, timeout=(5, 60))
            resp.raise_for_status()
            suffix = os.path.splitext(audio_url)[1] or ".mp3"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                for chunk in resp.iter_content(chunk_size=8192):
                    tf.write(chunk)
                return tf.name
        except Exception as e:
            logger.warning(f"⚠️ Download attempt {attempt} failed: {e}")
    raise RuntimeError(f"Failed to download audio after {retries} attempts")

# Lazily init the pyannote pipeline but with better error surfacing
_diari_pipeline = None
def _get_diarization_pipeline():
    global _diari_pipeline
    if _diari_pipeline is not None:
        return _diari_pipeline
    try:
        logger.info("⬇️ Loading pyannote speaker-diarization pipeline…")
        # Use `token=` (newer), and optional cache dir to avoid repeated downloads
        _diari_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            token=HF_TOKEN,
            cache_dir=os.getenv("HF_HOME") or "/root/.cache/huggingface"
        )
        logger.info("✅ Pyannote speaker-diarization pipeline loaded")
    except Exception as e:
        logger.error(
            "❌ Failed to load speaker-diarization pipeline. "
            "Common causes: missing HF token or gated access not approved; "
            "package/version mismatch; no ffmpeg; no network. "
            f"Raw error: {repr(e)}"
        )
        _diari_pipeline = None
    return _diari_pipeline

def transcribe(audio_url: str) -> list[dict]:
    logger.info(f"▶️ transcribe(): start for URL: {audio_url}")

    try:
        temp_path = _download_audio(audio_url)
        logger.info(f"✅ Downloaded audio to {temp_path}")
    except Exception as e:
        logger.error(f"❌ Audio download ultimately failed: {e}")
        raise

    words = []
    try:
        model = _get_model()
        segments, _ = model.transcribe(temp_path, beam_size=5, word_timestamps=True)
        words = [{"start": w.start, "end": w.end, "text": w.word}
                 for seg in segments for w in seg.words]
        logger.info(f"✅ Whisper ASR complete: {len(words)} words")
    except Exception as e:
        logger.error(f"❌ Whisper ASR failed: {e}")

    labeled = []
    pipeline = _get_diarization_pipeline()
    if pipeline and words:
        try:
            logger.info("▶️ Running speaker diarization…")
            diarization = pipeline({"audio": temp_path})
            # Build an index of turns to speed up lookups (optional)
            turns = list(diarization.itertracks(yield_label=True))
            for w in words:
                spk = next((label for (turn, _, label) in turns if turn.start <= w["start"] < turn.end), None)
                labeled.append({**w, "speaker": spk})
            logger.info(f"✅ Speaker diarization complete: {len(labeled)} words labeled")
        except Exception as e:
            logger.error(f"❌ Diarization failed at runtime: {e}")
            labeled = [{**w, "speaker": None} for w in words]
    else:
        if not pipeline:
            logger.warning("⚠️ Skipping diarization (pipeline unavailable)")
        labeled = [{**w, "speaker": None} for w in words]

    try:
        os.remove(temp_path)
        logger.info(f"🗑️ Temp file deleted: {temp_path}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to delete temp file: {e}")

    return labeled
