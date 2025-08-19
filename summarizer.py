# ======================== summarizer.py ========================
import os
from dotenv import load_dotenv
from typing import List, Optional

# Load .env for OPENAI_API_KEY / OPENAI_MODEL
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

# ---- SDK compatibility (openai<1.0 vs openai>=1.0) --------------------------
_NEW_SDK = False
_client = None
try:
    # openai>=1.0 style
    from openai import OpenAI  # type: ignore
    _client = OpenAI(api_key=OPENAI_API_KEY)
    _NEW_SDK = True
except Exception:
    # Fall back to openai<1.0
    import openai  # type: ignore
    openai.api_key = OPENAI_API_KEY
    _NEW_SDK = False

# ---- Tunables ---------------------------------------------------------------
# Prefer a current model; allow override via env
MODEL = os.getenv("OPENAI_MODEL", "").strip() or (
    "gpt-4o-mini"  # good default (cheap, strong); change if not available on your account
)
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))

# Chunking targets (approximate, word-based to avoid extra deps)
CHUNK_WORDS = int(os.getenv("SUMMARY_CHUNK_WORDS", "2200"))       # ~3k tokens text per chunk
CHUNK_SUMMARY_WORDS = int(os.getenv("SUMMARY_CHUNK_SUM_WORDS", "180"))  # per-chunk recap

# ----------------------------------------------------------------------------

def _split_into_word_chunks(text: str, chunk_words: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + chunk_words]) for i in range(0, len(words), chunk_words)]

def _chat(model: str, messages: list, max_tokens: int = 1200, temperature: float = TEMPERATURE) -> str:
    """
    Unified chat wrapper for both SDKs.
    """
    if _NEW_SDK:
        # openai>=1.0
        resp = _client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    else:
        # openai<1.0
        import openai  # type: ignore
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

# ---- Prompt helpers (force structure + takeaways) ---------------------------

def _format_requirements(target_words: int) -> str:
    return (
        "Return Markdown with these exact sections:\n"
        "### Overview\n"
        f"- A concise paragraph (~{max(120, target_words//2)}–{target_words} words) capturing the episode’s purpose and arc.\n"
        "### Key Takeaways\n"
        "- 5–10 bullet points with crisp, concrete takeaways (no fluff).\n"
        "### Insights & Analysis\n"
        "- 3–6 bullets explaining why the points matter, patterns, trade-offs, or implications.\n"
        "### Noteworthy Quotes\n"
        "- 2–5 short quotes (if present); otherwise write “None noted”. Keep them brief.\n"
        "### Outline by Topic\n"
        "- 3–7 bullets, each: **Section/Topic** — 1–2 sentences of what was covered.\n"
    )

def _summarize_chunk(chunk_text: str, target_words: int = CHUNK_SUMMARY_WORDS) -> str:
    system = (
        "You are a focused podcast summarizer. Summarize the chunk into a tight, factual recap "
        "and extract concrete takeaways. Prefer specificity over generalities."
    )
    user = (
        f"CHUNK START\n{chunk_text}\nCHUNK END\n\n"
        f"Write a structured recap (~{target_words} words max):\n"
        f"{_format_requirements(target_words)}"
    )
    return _chat(MODEL, [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
                 max_tokens=700)

def _compose_final_summary(chunk_summaries: List[str], max_words: int) -> str:
    joined = "\n\n---\n\n".join(chunk_summaries)
    system = (
        "You are a highly skilled podcast summarization assistant. "
        "Combine partial summaries of one episode into a single, coherent, non-repetitive summary. "
        "Merge duplicates, preserve chronology, and keep concrete details."
    )
    user = (
        "Using the partial summaries below, write the FINAL summary between {minw} and {maxw} words.\n"
        "Emphasize practical, non-obvious insights. Avoid generic phrases.\n\n"
        "{format}\n"
        "PARTIAL SUMMARIES:\n{chunks}\n\n"
        "FINAL SUMMARY:"
    ).format(
        minw=max(500, int(max_words * 0.9)),
        maxw=max_words,
        format=_format_requirements(max_words),
        chunks=joined,
    )
    return _chat(MODEL, [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
                 max_tokens=1400)

# ---- Public entrypoint ------------------------------------------------------

def summarize_with_openai(transcript: str, max_words: int = 800) -> str:
    """
    Summarize the given podcast transcript into structured Markdown:
    - Overview
    - Key Takeaways
    - Insights & Analysis
    - Noteworthy Quotes
    - Outline by Topic

    Uses a chunking strategy for long transcripts.
    """
    if not transcript or not transcript.strip():
        return "Transcript is empty."

    words = transcript.split()

    # Short transcripts: single pass with strict structure
    if len(words) <= CHUNK_WORDS:
        system = (
            "You are a highly skilled podcast summarization assistant. "
            "Produce a comprehensive, structured summary. Be specific and extract actionable insights."
        )
        user = (
            "Write a detailed summary between 500 and {maxw} words.\n"
            "Avoid generic phrasing; include concrete numbers/examples when present.\n\n"
            "{format}\n"
            "TRANSCRIPT START\n{tx}\nTRANSCRIPT END"
        ).format(maxw=max_words, format=_format_requirements(max_words), tx=transcript)
        return _chat(MODEL, [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                     max_tokens=1400)

    # Long transcripts: chunk → per-chunk summaries → final composition
    chunks = _split_into_word_chunks(transcript, CHUNK_WORDS)

    chunk_summaries: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        try:
            chunk_summaries.append(_summarize_chunk(chunk, target_words=CHUNK_SUMMARY_WORDS))
        except Exception as e:
            chunk_summaries.append(f"(Failed to summarize chunk {i}: {e})")

    # Optional folding if too long to combine directly (rare)
    joined_len = sum(len(s) for s in chunk_summaries)
    if joined_len > 12000:
        folded: List[str] = []
        pack: List[str] = []
        char_budget = 6000
        cur = 0
        for s in chunk_summaries:
            if cur + len(s) > char_budget and pack:
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

    return _compose_final_summary(chunk_summaries, max_words=max_words)
