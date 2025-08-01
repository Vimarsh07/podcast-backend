# ======================== summarizer.py ========================
import os
from dotenv import load_dotenv
import openai
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def summarize_with_openai(transcript_text: str) -> str:
    prompt = ("Summarize the following podcast transcript:\n\n" + transcript_text[:6000])
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return resp.choices[0].message["content"].strip()