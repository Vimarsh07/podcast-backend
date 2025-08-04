# summarizer.py

import os
import openai
from dotenv import load_dotenv

load_dotenv()  # loads OPENAI_API_KEY from .env

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("Missing OPENAI_API_KEY")

def summarize_with_openai(transcript: str) -> str:
    """
    Summarize the given podcast transcript in detail, producing a 500–1000 word summary
    that highlights the main topics, key insights, and any notable quotes.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a highly skilled podcast summarization assistant. "
                "Your task is to read a full transcript of a podcast episode and "
                "produce a comprehensive, coherent, and engaging summary. "
                "Focus on the main topics, important arguments or stories, "
                "and any memorable quotes or moments."
            )
        },
        {
            "role": "user",
            "content": (
                "Please provide a detailed summary of the following podcast episode transcript. "
                "Your summary should be between 500 and 1000 words in length, "
                "and should cover:\n"
                "  • The overall theme and purpose of the episode\n"
                "  • Each major section or discussion point, with brief explanations\n"
                "  • Key takeaways, insights, and actionable advice (if any)\n"
                "  • Noteworthy quotes or highlights\n\n"
                f"Transcript:\n\n{transcript}"
            )
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3,
        max_tokens=1200,
    )

    return response.choices[0].message.content.strip()
