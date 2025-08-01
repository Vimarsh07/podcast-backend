# ======================== transcriber.py ========================
import os
import logging
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# Logging
logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# Initialize Pyannote
HF_TOKEN = (os.getenv("HF_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN"))
try:
    if HF_TOKEN:
        diari_pipeline = Pipeline.from_pretrained(
            "pyannote/segmentation", use_auth_token=HF_TOKEN)
    else:
        diari_pipeline = Pipeline.from_pretrained(
            "pyannote/segmentation", use_auth_token=True)
    logging.info("✅ Pyannote segmentation pipeline loaded")
except Exception as e:
    logging.error(f"❌ Failed to load pyannote pipeline: {e}")
    diari_pipeline = None


def transcribe(audio_path: str):
    logging.info(f"Starting transcription for: {audio_path}")
    # choose device
    if torch.cuda.is_available():
        try:
            model = WhisperModel("base", device="cuda", compute_type="float16")
            use_device, precision = "cuda", "float16"
        except Exception:
            model = WhisperModel("base", device="cpu", compute_type="float32")
            use_device, precision = "cpu", "float32"
    else:
        model = WhisperModel("base", device="cpu", compute_type="float32")
        use_device, precision = "cpu", "float32"
    logging.info(f"Using device={use_device}, precision={precision}")

    # ASR
    segments, info = model.transcribe(
        audio_path, beam_size=5, word_timestamps=True)
    raw = []
    for seg in segments:
        for w in seg.words:
            raw.append({"start": w.start, "end": w.end, "text": w.word})
    logging.info(f"ASR complete: {len(raw)} word segments")

    # Diarization & labeling
    labeled = []
    if diari_pipeline:
        try:
            dia = diari_pipeline({"audio": audio_path})
            for seg in raw:
                spk = None
                for turn, _, label in dia.itertracks(yield_label=True):
                    if seg["start"] >= turn.start < seg["start"] < turn.end:
                        spk = label
                        break
                labeled.append({**seg, "speaker": spk})
            logging.info(f"Labeled {len(labeled)} segments with speakers")
        except Exception:
            logging.warning("Diarization failed, no speaker tags")
            labeled = [{**seg, "speaker": None} for seg in raw]
    else:
        logging.warning("No diarization pipeline; skipping speaker labels")
        labeled = [{**seg, "speaker": None} for seg in raw]

    return labeled
