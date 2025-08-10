# transcriber.py
import os
import json
import logging
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from cog import BasePredictor, Input, Path

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger("transcriber")


# ----------------------------
# Helpers
# ----------------------------
def _download_audio(audio_url: str, retries: int = 3, timeout: Tuple[int, int] = (5, 60)) -> str:
    """Download audio to a temp file and return the local path."""
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(audio_url, stream=True, timeout=timeout)
            resp.raise_for_status()
            suffix = os.path.splitext(audio_url)[1] or ".mp3"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                for chunk in resp.iter_content(chunk_size=8192):
                    tf.write(chunk)
                return tf.name
        except Exception as e:
            last_err = e
            logger.warning(f"⚠️ Download attempt {attempt}/{retries} failed: {e}")
    raise RuntimeError(f"Failed to download audio after {retries} attempts: {last_err}")


def _annotation_to_turns(ann: Annotation) -> List[Tuple[Segment, str]]:
    """Flatten pyannote Annotation into list of (Segment, speaker_label)."""
    turns: List[Tuple[Segment, str]] = []
    # itertracks yields ((Segment, Track), label)
    for (segment, _track), label in ann.itertracks(yield_label=True):
        turns.append((segment, str(label)))
    # sort by time
    turns.sort(key=lambda t: (float(t[0].start), float(t[0].end)))
    return turns


def _assign_speaker_for_word(
    t: float,
    turns: List[Tuple[Segment, str]],
) -> Optional[str]:
    """Find speaker label for a time t (seconds) using turns."""
    # Primary: containment
    for segment, label in turns:
        if segment.start <= t < segment.end:
            return label
    # Fallback: nearest segment center
    if not turns:
        return None
    nearest = min(
        turns, key=lambda s: abs(((s[0].start + s[0].end) / 2.0) - t)
    )
    return nearest[1]


# ----------------------------
# Cog Predictor
# ----------------------------
class Predictor(BasePredictor):
    """
    Replicate/Cog entrypoint:
    - setup(): load Faster-Whisper + pyannote once
    - predict(): accept URL or file, run ASR then diarization, return JSON
    """

    def setup(self) -> None:
        # ---- HF token (required for pyannote 3.1) ----
        self.hf_token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HF_HUB_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
        )
        if not self.hf_token:
            raise RuntimeError(
                "Missing Hugging Face token. Set HF_TOKEN in Replicate model settings, "
                "and ensure access to 'pyannote/speaker-diarization-3.1' is granted."
            )

        # ---- Device for Faster-Whisper ----
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "float32"
        logger.info(f"Device: {self.device} | compute_type: {self.compute_type}")

        # ---- Load Faster-Whisper model once ----
        # You can change "base" to "small", "medium", etc. per your cost/quality needs.
        self.asr: WhisperModel = WhisperModel("base", device=self.device, compute_type=self.compute_type)
        logger.info("✅ Faster-Whisper loaded")

        # ---- Load pyannote diarization pipeline once ----
        # Prefer the 3.1 pipeline (gated). Cache dir helps warm start if reused.
        self.hf_cache = os.getenv("HF_HOME") or "/root/.cache/huggingface"
        model_id = os.getenv("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")
        self.diar: Pipeline = Pipeline.from_pretrained(
            model_id,
            use_auth_token=self.hf_token,  # older versions: token=self.hf_token
            cache_dir=self.hf_cache,
        )
        logger.info(f"✅ Pyannote pipeline loaded: {model_id}")

        # Optional: default tunables
        self.diar_params: Dict[str, Any] = {
            # e.g., "num_speakers": 2
            # e.g., "segmentation": {"min_duration_on": 0.0, "threshold": 0.5},
            # e.g., "clustering": {"method": "pool", "threshold": 0.715},
        }

    def predict(
        self,
        audio_url: Optional[str] = Input(
            description="Publicly accessible URL to an audio file (mp3/wav/flac/m4a).",
            default=None,
        ),
        audio_file: Optional[Path] = Input(
            description="Or upload a local audio file instead of a URL.",
            default=None,
        ),
        num_speakers: Optional[int] = Input(
            description="If known, force the number of speakers (pyannote hint).",
            default=None,
        ),
        word_timestamps: bool = Input(
            description="Return word-level timestamps from ASR.",
            default=True,
        ),
        beam_size: int = Input(
            description="Decoding beam size for Faster-Whisper.",
            default=5,
        ),
    ) -> Dict[str, Any]:
        """
        Run Faster-Whisper ASR + pyannote diarization.
        Returns JSON with words (and speaker labels) + a tiny summary.
        """
        if not audio_url and not audio_file:
            raise ValueError("Provide either 'audio_url' or 'audio_file'.")

        # ---- Get local path ----
        tmp_path: Optional[str] = None
        input_path: str
        if audio_url:
            tmp_path = _download_audio(audio_url)
            input_path = tmp_path
            logger.info(f"✅ Downloaded: {input_path}")
        else:
            input_path = str(audio_file)
            logger.info(f"✅ Using uploaded file: {input_path}")

        # ---- ASR ----
        words: List[Dict[str, Any]] = []
        try:
            segments, _info = self.asr.transcribe(
                input_path,
                beam_size=beam_size,
                word_timestamps=word_timestamps,
            )

            if word_timestamps:
                # flatten word-level
                for seg in segments:
                    for w in seg.words:
                        words.append(
                            {
                                "start": float(w.start),
                                "end": float(w.end),
                                "text": w.word,
                            }
                        )
            else:
                # segment-level only
                for seg in segments:
                    words.append(
                        {
                            "start": float(seg.start),
                            "end": float(seg.end),
                            "text": seg.text or "",
                        }
                    )

            logger.info(f"✅ ASR complete: {len(words)} items")
        except Exception as e:
            # Bubble up—caller needs to know ASR failed
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            raise RuntimeError(f"Whisper ASR failed: {e}")

        # ---- Diarization ----
        diar_kwargs = dict(self.diar_params)
        if num_speakers is not None:
            diar_kwargs["num_speakers"] = int(num_speakers)

        try:
            ann: Annotation = self.diar({"audio": input_path}, **diar_kwargs)
            turns = _annotation_to_turns(ann)

            # Assign speakers to words
            labeled: List[Dict[str, Any]] = []
            for w in words:
                mid = (w["start"] + w["end"]) / 2.0
                spk = _assign_speaker_for_word(mid, turns)
                labeled.append({**w, "speaker": spk})

            logger.info(f"✅ Diarization complete: {len(labeled)} labeled items")
        except Exception as e:
            logger.error(f"❌ Diarization failed: {e}")
            labeled = [{**w, "speaker": None} for w in words]

        # ---- Cleanup ----
        if tmp_path:
            try:
                os.remove(tmp_path)
                logger.info(f"🗑️ Deleted temp file: {tmp_path}")
            except Exception as e:
                logger.warning(f"⚠️ Temp cleanup failed: {e}")

        # ---- Response ----
        speakers = sorted(list({w["speaker"] for w in labeled if w["speaker"]}))
        return {
            "device": self.device,
            "num_items": len(labeled),
            "estimated_speakers": len(speakers),
            "speakers": speakers,
            "words": labeled,
        }
