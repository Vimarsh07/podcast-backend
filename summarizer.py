# ======================== summarizer.py ========================
import os
import math
import openai
from dotenv import load_dotenv
from typing import List

load_dotenv()  # loads OPENAI_API_KEY from .env

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Missing OPENAI_API_KEY")

# --- Tunables ---
MODEL = "gpt-3.5-turbo"    # if you have access to larger-context models, you can swap here
TEMPERATURE = 0.3
# Roughly: 4 chars ≈ 1 token, 1 token ≈ 0.75 words (very rough).
# We'll chunk by words to avoid any external deps; be conservative:
CHUNK_WORDS = 2200         # ~3k tokens worth of text per chunk (safe for 16k context models)
CHUNK_SUMMARY_WORDS = 180   # keep chunk summaries tight; makes the final pass robust


def _split_into_word_chunks(text: str, chunk_words: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), chunk_words):
        chunk = " ".join(words[i : i + chunk_words])
        chunks.append(chunk)
    return chunks


def _chat(model: str, messages: list, max_tokens: int = 1200, temperature: float = TEMPERATURE) -> str:
    """
    Wrapper around ChatCompletion. If you're on openai<1.0, this works as-is.
    If you've upgraded to openai>=1.0, migrate to the new client:
      from openai import OpenAI
      client = OpenAI()
      client.chat.completions.create(model=..., messages=..., ...)
    """
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _summarize_chunk(chunk_text: str, target_words: int = CHUNK_SUMMARY_WORDS) -> str:
    system = (
        "You are a focused summarizer. Read the chunk of a podcast transcript and produce a concise, "
        f"coherent summary (~{target_words} words) highlighting main ideas, arguments, and any notable facts."
    )
    user = f"Chunk transcript:\n\n{chunk_text}\n\nSummary (~{target_words} words):"
    return _chat(MODEL, [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
                 max_tokens=600)


def _compose_final_summary(chunk_summaries: List[str], max_words: int) -> str:
    joined = "\n\n---\n\n".join(chunk_summaries)
    system = (
        "You are a highly skilled podcast summarization assistant. "
        "Given several partial summaries of different parts of one episode, "
        "write a single comprehensive, coherent summary."
    )
    user = (
        "Write a final summary between {minw} and {maxw} words that covers:\n"
        "• Overall theme and purpose\n"
        "• Major sections / discussion points\n"
        "• Key takeaways and insights\n"
        "• Noteworthy quotes or highlights (if present)\n\n"
        "Partial summaries:\n\n{chunks}\n\nFinal summary:"
    ).format(minw=max(500, int(max_words*0.9)), maxw=max_words, chunks=joined)

    # Leave headroom in max_tokens; we control length via instruction, not hard token limit.
    return _chat(MODEL, [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
                 max_tokens=1200)


def summarize_with_openai(transcript: str, max_words: int = 800) -> str:
    """
    Summarize the given podcast transcript.
    - If short enough: single-pass detailed summary
    - If long: chunk -> per-chunk summary -> final stitched summary
    """
    if not transcript or not transcript.strip():
        return "Transcript is empty."

    words = transcript.split()
    # If small enough, do one pass.
    if len(words) <= CHUNK_WORDS:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a highly skilled podcast summarization assistant. "
                    "Your task is to read a full transcript of a podcast episode and "
                    "produce a comprehensive, coherent, and engaging summary. "
                    "Focus on the main topics, important arguments or stories, "
                    "and any memorable quotes or moments."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Provide a detailed summary between 500 and {maxw} words that covers:\n"
                    "• The overall theme and purpose of the episode\n"
                    "• Each major section or discussion point\n"
                    "• Key takeaways, insights, and actionable advice (if any)\n"
                    "• Noteworthy quotes or highlights\n\n"
                    "Transcript:\n\n{tx}".format(maxw=max_words, tx=transcript)
                ),
            },
        ]
        # give the model breathing room
        return _chat(MODEL, messages, max_tokens=1200)

    # Otherwise: chunk it
    chunks = _split_into_word_chunks(transcript, CHUNK_WORDS)

    # Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks, start=1):
        try:
            summary = _summarize_chunk(chunk, target_words=CHUNK_SUMMARY_WORDS)
        except Exception as e:
            summary = f"(Failed to summarize chunk {i}: {e})"
        chunk_summaries.append(summary)

    # If there are too many chunk summaries to fit, fold them once
    # (rare, but protects against exceedingly long episodes).
    joined_len = sum(len(s) for s in chunk_summaries)
    if joined_len > 12000:  # rough guard; adjust if you switch models
        folded = []
        pack = []
        char_budget = 6000
        cur = 0
        for s in chunk_summaries:
            if cur + len(s) > char_budget and pack:
                # summarize this pack into one
                pack_text = "\n\n---\n\n".join(pack)
                folded.append(_summarize_chunk(pack_text, target_words=CHUNK_SUMMARY_WORDS * 2))
                pack = []
                cur = 0
            pack.append(s)
            cur += len(s)
        if pack:
            pack_text = "\n\n---\n\n".join(pack)
            folded.append(_summarize_chunk(pack_text, target_words=CHUNK_SUMMARY_WORDS * 2))
        chunk_summaries = folded

    # Compose final summary
    return _compose_final_summary(chunk_summaries, max_words=max_words)
