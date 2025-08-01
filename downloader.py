# ======================== downloader.py ========================
import requests
from pathlib import Path
import os


def download_audio(url: str, output_filename: str) -> str:
    """Download an audio URL to the project's directory"""
    project_dir = Path(os.path.abspath(os.path.dirname(__file__)))
    output_path = project_dir / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True)
    with open(output_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return str(output_path)