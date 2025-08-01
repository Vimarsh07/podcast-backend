
# ======================== downloader.py ========================
import requests
from pathlib import Path


def download_audio(url: str, output_path: str) -> str:
    """Download an audio URL to the given path"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True)
    with open(output_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return output_path
