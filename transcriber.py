# transcriber.py

import os
import logging
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# ─── Initialize Speaker-Diarization Pipeline ────────────────────────────────
HF_TOKEN = (
    os.getenv("HF_HUB_TOKEN")
    or os.getenv("HUGGINGFACE_TOKEN")
    or os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
try:
    diari_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=HF_TOKEN or True
    )
    logging.info("✅ Pyannote speaker-diarization pipeline loaded")
except Exception as e:
    logging.error(f"❌ Failed to load speaker-diarization pipeline: {e}")
    diari_pipeline = None


def transcribe(audio_path: str):
    logging.info(f"Starting transcription for: {audio_path}")

    # ─── ASR Model Load ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        try:
            model = WhisperModel("base", device="cuda", compute_type="float16")
            device = "cuda"
        except Exception:
            model = WhisperModel("base", device="cpu", compute_type="float32")
            device = "cpu"
    else:
        model = WhisperModel("base", device="cpu", compute_type="float32")
        device = "cpu"
    logging.info(f"Using device={device}")

    # ─── Run ASR (word timestamps) ─────────────────────────────────────────────
    segments, _ = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    words = [
        {"start": w.start, "end": w.end, "text": w.word}
        for seg in segments for w in seg.words
    ]
    logging.info(f"ASR complete: {len(words)} words")

    # ─── Run Speaker Diarization ───────────────────────────────────────────────
    if diari_pipeline:
        try:
            logging.info("Running speaker-diarization…")
            diarization = diari_pipeline({"audio": audio_path})
        except Exception as e:
            logging.warning(f"Diarization failed ({e}); skipping speaker tags")
            return [{**w, "speaker": None} for w in words]
    else:
        logging.warning("No diarization pipeline; skipping speaker tags")
        return [{**w, "speaker": None} for w in words]

    # ─── Assign Speakers to Words ──────────────────────────────────────────────
    labeled = []
    for w in words:
        spk = None
        for turn, _, label in diarization.itertracks(yield_label=True):
            if turn.start <= w["start"] < turn.end:
                spk = label
                break
        labeled.append({**w, "speaker": spk})

    logging.info(f"Assigned speaker labels to {len(labeled)} words")
    return labeled
