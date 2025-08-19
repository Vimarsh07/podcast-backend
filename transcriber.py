# transcriber.py
import os
import math
import logging
import tempfile
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from cog import BasePredictor, Input, Path, Secret

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from huggingface_hub import login

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
            suffix = os.path.splitext(audio_url.split("?")[0])[1] or ".mp3"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                for chunk in resp.iter_content(chunk_size=8192):
                    tf.write(chunk)
                return tf.name
        except Exception as e:
            last_err = e
            logger.warning(f"⚠️ Download attempt {attempt}/{retries} failed: {e}")
    raise RuntimeError(f"Failed to download audio after {retries} attempts: {last_err}")

def _ffmpeg_convert_to_wav16k_mono(src_path: str) -> str:
    """Convert any input to 16kHz mono WAV for consistent, smaller-footprint processing."""
    out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_path

def _ffprobe_duration_seconds(path: str) -> float:
    """Get media duration using ffprobe."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", path]
    res = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(res.stdout.strip())
    except Exception:
        return 0.0

def _chunk_audio_wav16k(src_wav: str, chunk_seconds: int) -> List[Tuple[str, float]]:
    """Split a WAV16k mono file into chunks of chunk_seconds. Returns list of (chunk_path, chunk_start_sec)."""
    total = _ffprobe_duration_seconds(src_wav)
    if total <= 0:
        return [(src_wav, 0.0)]
    chunks: List[Tuple[str, float]] = []
    num_chunks = max(1, math.ceil(total / chunk_seconds))
    for i in range(num_chunks):
        start = i * chunk_seconds
        dur = min(chunk_seconds, max(0.0, total - start))
        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        cmd = ["ffmpeg", "-y", "-ss", str(start), "-t", str(dur), "-i", src_wav, "-ac", "1", "-ar", "16000", out]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        chunks.append((out, float(start)))
    return chunks

def _annotation_to_turns(ann: Annotation) -> List[Tuple[Segment, str]]:
    """Flatten pyannote Annotation into list of (Segment, speaker_label).
    Handles both (segment, track, label) and ((segment, track), label) yields.
    """
    turns: List[Tuple[Segment, str]] = []
    for item in ann.itertracks(yield_label=True):
        if isinstance(item, tuple) and len(item) == 3:
            segment, _track, label = item  # newer: (segment, track, label)
        else:
            (segment, _track), label = item  # older: ((segment, track), label)
        turns.append((segment, str(label)))
    turns.sort(key=lambda t: (float(t[0].start), float(t[0].end)))
    return turns

def _assign_speaker_for_word(t: float, turns: List[Tuple[Segment, str]]) -> Optional[str]:
    """Find speaker label for a time t (seconds) using turns."""
    for segment, label in turns:
        if segment.start <= t < segment.end:
            return label
    if not turns:
        return None
    nearest = min(
        turns,
        key=lambda s: abs(((s[0].start + s[0].end) / 2.0) - t)
    )
    return nearest[1]

# ----------------------------
# Cog Predictor
# ----------------------------
class Predictor(BasePredictor):
    """
    ASR (Faster-Whisper) on GPU by default with CPU fallback per chunk.
    Diarization (pyannote) on CPU by default.
    Chunked processing to cap peak memory.
    """

    def setup(self) -> None:
        # Be stingy with threads (helps stability on small workers)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        # Lazy ASR (created per requested device)
        self._asr: Optional[WhisperModel] = None
        self._asr_device: Optional[str] = None

        # Lazy diarizer
        self._diar_pipeline: Optional[Pipeline] = None
        self._diar_model_id: Optional[str] = None
        self._diar_device: str = "cpu"
        self.hf_cache = os.getenv("HF_HOME") or "/root/.cache/huggingface"

    # ---------- ASR (lazy per device) ----------
    def _get_asr(self, device: str) -> WhisperModel:
        device = device.lower()
        if device not in {"cpu", "cuda"}:
            device = "cpu"
        if self._asr is None or self._asr_device != device:
            model_name = os.getenv("WHISPER_MODEL", "tiny")  # T4-safe default; override via env
            compute = "int8_float16" if (device == "cuda" and torch.cuda.is_available()) else "float32"
            logger.info(f"Loading Faster-Whisper '{model_name}' on {device} (compute={compute})")
            self._asr = WhisperModel(
                model_name,
                device=device,
                compute_type=compute,
                device_index=0,
                cpu_threads=2,
                num_workers=1,
            )
            self._asr_device = device
            logger.info("✅ Faster-Whisper ready")
        return self._asr

    # ---------- Diarizer (lazy) ----------
    def _ensure_diarizer(self, hf_token: str, model_id: str, device: str) -> Pipeline:
        """Login to HF and (lazily) load/cached pyannote pipeline with the provided token."""
        login(token=hf_token)
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token

        if self._diar_pipeline is None or self._diar_model_id != model_id or self._diar_device != device:
            logger.info(f"Loading pyannote pipeline: {model_id} on {device}")
            try:
                self._diar_pipeline = Pipeline.from_pretrained(
                    model_id,
                    use_auth_token=hf_token,
                    cache_dir=self.hf_cache,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load '{model_id}'. Ensure this token has accepted access "
                    f"to required gated repos (e.g., 'pyannote/segmentation-3.0'). Original: {e}"
                )
            self._diar_model_id = model_id
            self._diar_device = device

            # Move to target device (CPU default for stability)
            try:
                target = torch.device("cuda") if device == "cuda" and torch.cuda.is_available() else torch.device("cpu")
                self._diar_pipeline.to(target)
            except Exception:
                pass

            logger.info(f"✅ Pyannote pipeline ready: {model_id} ({device})")

        return self._diar_pipeline

    # ---------- Per-chunk workers ----------
    def _whisper_transcribe_chunk(
        self,
        asr: WhisperModel,
        chunk_path: str,
        beam_size: int,
        word_timestamps: bool,
        language: Optional[str],
    ) -> List[Dict[str, Any]]:
        """ASR on a single chunk; CUDA OOM => retry on CPU."""
        kwargs = {
            "beam_size": beam_size,
            "word_timestamps": word_timestamps,
            "vad_filter": True,
        }
        if language:
            kwargs["language"] = language

        try:
            segments, _info = asr.transcribe(chunk_path, **kwargs)
        except Exception as e:
            msg = str(e)
            if ("CUDA out of memory" in msg) or ("cuda" in msg.lower() and "oom" in msg.lower()):
                logger.warning("⚠️ CUDA OOM during ASR, retrying chunk on CPU…")
                cpu_asr = self._get_asr("cpu")
                segments, _info = cpu_asr.transcribe(chunk_path, **kwargs)
            else:
                raise

        words: List[Dict[str, Any]] = []
        if word_timestamps:
            for seg in segments:
                for w in seg.words:
                    words.append({"start": float(w.start), "end": float(w.end), "text": w.word})
        else:
            for seg in segments:
                words.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text or ""})
        return words

    def _diarize_chunk(
        self,
        diar: Pipeline,
        chunk_path: str,
        diar_kwargs: Dict[str, Any],
    ) -> List[Tuple[Segment, str]]:
        """Diarize one chunk; if CUDA OOM, force CPU and retry."""
        try:
            ann: Annotation = diar({"audio": chunk_path}, **diar_kwargs)
        except Exception as e:
            msg = str(e)
            if ("CUDA out of memory" in msg) or ("cuda" in msg.lower() and "oom" in msg.lower()):
                logger.warning("⚠️ CUDA OOM during diarization, retrying on CPU…")
                try:
                    diar.to(torch.device("cpu"))
                except Exception:
                    pass
                ann = diar({"audio": chunk_path}, **diar_kwargs)
            else:
                raise
        return _annotation_to_turns(ann)

    # ---------- Predict ----------
    def predict(
        self,
        audio_url: Optional[str] = Input(
            description="Public URL to an audio file (mp3/wav/flac/m4a).",
            default=None,
        ),
        audio_file: Optional[Path] = Input(
            description="Or upload a local audio file instead of a URL.",
            default=None,
        ),
        hf_token: Secret = Input(
            description="Hugging Face token (required for pyannote 3.x / gated models).",
        ),
        diarization_model: str = Input(
            description="Pyannote pipeline repo id.",
            default="pyannote/speaker-diarization-3.1",
        ),
        num_speakers: Optional[int] = Input(
            description="If known, force the number of speakers (pyannote hint).",
            default=None,
        ),
        # T4-safe defaults for speed+stability
        word_timestamps: bool = Input(
            description="Return word-level timestamps from ASR.",
            default=False,
        ),
        beam_size: int = Input(
            description="Decoding beam size for Faster-Whisper.",
            default=1,
        ),
        chunk_seconds: int = Input(
            description="Process audio in chunks of this many seconds (reduces peak memory).",
            default=180,  # 3 minutes
        ),
        asr_device: str = Input(
            description="Device for ASR: 'cuda' (default) or 'cpu'.",
            default="cuda",
        ),
        diar_device: str = Input(
            description="Device for diarization: 'cpu' (default, safest) or 'cuda'.",
            default="cpu",
        ),
        language: Optional[str] = Input(
            description="Language hint like 'en', 'es'. If set, skips autodetect.",
            default="en",
        ),
    ) -> Dict[str, Any]:
        """
        Chunked Faster-Whisper ASR (GPU with CPU fallback) + pyannote diarization (CPU).
        Returns JSON with words (and speaker labels).
        """
        if not audio_url and not audio_file:
            raise ValueError("Provide either 'audio_url' or 'audio_file'.")

        # Paths to clean up
        tmp_download: Optional[str] = None
        tmp_wav16k: Optional[str] = None
        tmp_chunks: List[str] = []

        try:
            # ---- Get local path ----
            if audio_url:
                tmp_download = _download_audio(audio_url)
                input_path = tmp_download
                logger.info(f"✅ Downloaded: {input_path}")
            else:
                input_path = str(audio_file)
                logger.info(f"✅ Using uploaded file: {input_path}")

            # ---- Convert to WAV 16k mono ----
            tmp_wav16k = _ffmpeg_convert_to_wav16k_mono(input_path)

            # ---- Chunk audio ----
            chunk_len = max(90, int(chunk_seconds))  # minimum 90s
            chunks = _chunk_audio_wav16k(tmp_wav16k, chunk_len)
            tmp_chunks = [p for p, _ in chunks]
            total_dur = _ffprobe_duration_seconds(tmp_wav16k)
            mm = int(total_dur // 60)
            ss = total_dur - 60 * mm
            logger.info(f"Processing audio with duration {mm:02d}:{ss:06.3f} in {len(chunks)} chunk(s)")

            # ---- Build diar kwargs (only supported keys) ----
            diar_kwargs: Dict[str, Any] = {}
            if num_speakers is not None:
                diar_kwargs["num_speakers"] = int(num_speakers)

            # ---- Prepare ASR (lazy per device) ----
            asr = self._get_asr(asr_device)

            # ---- Process each chunk sequentially ----
            all_words: List[Dict[str, Any]] = []
            all_turns: List[Tuple[Segment, str]] = []
            diar = None  # lazy init on first diarization

            for chunk_path, offset in chunks:
                # ASR
                words = self._whisper_transcribe_chunk(asr, chunk_path, beam_size, word_timestamps, language)
                for w in words:
                    w["start"] += offset
                    w["end"] += offset
                all_words.extend(words)

                if torch.cuda.is_available() and asr_device.lower() == "cuda":
                    torch.cuda.empty_cache()

                # Diarization (initialize when needed to keep peak mem low)
                if diar is None:
                    diar = self._ensure_diarizer(hf_token.get_secret_value(), diarization_model, diar_device.lower())
                turns = self._diarize_chunk(diar, chunk_path, diar_kwargs)
                for seg, _spk in turns:
                    seg.start += offset
                    seg.end += offset
                all_turns.extend(turns)

                if torch.cuda.is_available() and (asr_device.lower() == "cuda" or diar_device.lower() == "cuda"):
                    torch.cuda.empty_cache()

            # ---- Assign speakers to words ----
            labeled: List[Dict[str, Any]] = []
            for w in all_words:
                mid = (w["start"] + w["end"]) / 2.0
                spk = _assign_speaker_for_word(mid, all_turns)
                labeled.append({**w, "speaker": spk})

            speakers = sorted(list({w["speaker"] for w in labeled if w["speaker"]}))
            return {
                "device_asr": self._asr_device or "cpu",
                "device_diar": self._diar_device,
                "num_items": len(labeled),
                "estimated_speakers": len(speakers),
                "speakers": speakers,
                "words": labeled,
            }

        finally:
            # Cleanup temp files
            def _rm(p: Optional[str]):
                try:
                    if p and os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            _rm(tmp_download)
            _rm(tmp_wav16k)
            for p in tmp_chunks:
                _rm(p)
