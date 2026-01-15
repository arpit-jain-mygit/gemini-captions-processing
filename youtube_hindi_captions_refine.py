#!/usr/bin/env python3
import os
import re
import sys
import time
import json
import logging
import smtplib
import requests
import unicodedata
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound
from google import genai

# =========================================================
# CONFIG
# =========================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")

EMAIL_FROM = os.environ.get("EMAIL_FROM")
EMAIL_TO = os.environ.get("EMAIL_TO")
EMAIL_APP_PASSWORD = os.environ.get("EMAIL_APP_PASSWORD")

MODEL_NAME = "gemini-2.5-flash"
STATE_FILE = "processed_videos.json"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("yt-captions")

# =========================================================
# PROMPT (UNCHANGED)
# =========================================================
AUDIO_PROMPT = """
Role:
You are a senior Jain Agam scholar, Prakrit language expert, and professional Hindi transcriber.

Primary Task:
Produce a faithful, verbatim transcription of the attached Jain pravachan audio in Unicode Devanagari.

Critical Instructions (follow strictly):

1. Transcription Accuracy
- Transcribe what is spoken, but resolve obvious ASR ambiguities using Jain religious context.
- If two Hindi words sound similar, choose the one that makes doctrinal and grammatical sense.

2. Jain Mantra & Prakrit Normalization
- Normalize all Jain mantras to their canonical Prakrit forms.

3. Repetition Handling
- Do NOT repeat mantras unless clearly repeated.

4. Spoken Pauses vs Punctuation
- Do NOT insert punctuation based on pauses.

5. Spoken Fillers
- Remove filler noise.

6. Style Preservation
- Preserve oral teaching style.
- Do NOT summarize or interpret.

Output Requirements:
- Clean Unicode Devanagari text
- No English, no markdown
"""

# =========================================================
# FILENAME FIX (IMPORTANT)
# =========================================================
def safe_filename(title: str, max_len: int = 180) -> str:
    """
    Unicode-safe, cross-platform filename generator.
    Preserves Hindi + English.
    """
    title = unicodedata.normalize("NFKC", title)
    title = title.replace("/", "-").replace("\\", "-")
    title = re.sub(r'[<>:"|?*]', "", title)
    title = re.sub(r"\s+", "_", title)
    return title.strip("_")[:max_len]

# =========================================================
# HELPERS
# =========================================================
def extract_video_id(url: str) -> str | None:
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

def extract_playlist_id(url: str) -> str | None:
    m = re.search(r"list=([a-zA-Z0-9_-]+)", url)
    return m.group(1) if m else None

def yt_api(endpoint, params):
    r = requests.get(
        f"https://www.googleapis.com/youtube/v3/{endpoint}",
        params=params,
        timeout=20,
    )
    r.raise_for_status()
    return r.json()

def get_video_title(video_id: str) -> str:
    data = yt_api(
        "videos",
        {
            "part": "snippet",
            "id": video_id,
            "key": YOUTUBE_API_KEY,
        },
    )
    return data["items"][0]["snippet"]["title"]

def get_playlist_video_ids(pid: str) -> list[str]:
    ids = []
    page = None
    while True:
        data = yt_api(
            "playlistItems",
            {
                "part": "contentDetails",
                "playlistId": pid,
                "maxResults": 50,
                "pageToken": page,
                "key": YOUTUBE_API_KEY,
            },
        )
        for it in data["items"]:
            ids.append(it["contentDetails"]["videoId"])
        page = data.get("nextPageToken")
        if not page:
            break
    return ids

def load_state():
    if Path(STATE_FILE).exists():
        return json.loads(Path(STATE_FILE).read_text())
    return {}

def save_state(state):
    Path(STATE_FILE).write_text(
        json.dumps(state, indent=2, ensure_ascii=False)
    )

# =========================================================
# CORE LOGIC
# =========================================================
def fetch_hindi_captions(video_id: str) -> str:
    api = YouTubeTranscriptApi()
    try:
        t = api.list(video_id).find_manually_created_transcript(["hi"])
    except NoTranscriptFound:
        t = api.list(video_id).find_generated_transcript(["hi"])
    lines = t.fetch()
    return "\n".join(x.text for x in lines)

def refine_with_gemini(text: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[AUDIO_PROMPT, text],
        config={"temperature": 0.1},
    )
    return resp.text or ""

# =========================================================
# EMAIL
# =========================================================
def send_email(files: list[Path], url: str):
    if not files:
        logger.warning("No outputs to email")
        return

    logger.info("Preparing email")

    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = "YouTube Hindi Captions – Refined Output"

    body = [
        "YouTube URL:",
        url,
        "",
        "Refined caption files:",
    ]
    for f in files:
        body.append(f"- {f.name}")

    msg.set_content("\n".join(body))

    for f in files:
        msg.add_attachment(
            f.read_bytes(),
            maintype="text",
            subtype="plain",
            filename=f.name,
        )

    logger.info("Sending email...")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
        s.send_message(msg)

    logger.info("Email sent successfully")

# =========================================================
# MAIN
# =========================================================
def main(url: str):
    start = time.time()
    state = load_state()
    outputs = []

    pid = extract_playlist_id(url)
    vid = extract_video_id(url)

    if pid:
        video_ids = get_playlist_video_ids(pid)
    elif vid:
        video_ids = [vid]
    else:
        raise ValueError("Invalid YouTube URL")

    logger.info(f"Total videos: {len(video_ids)}")

    for v in video_ids:
        if state.get(v):
            logger.info(f"Skipping: {v}")
            continue

        try:
            title = get_video_title(v)
            fname = safe_filename(title) + ".txt"
            out_file = Path(OUTPUT_DIR) / fname

            logger.info(f"▶ {v} | captions")
            captions = fetch_hindi_captions(v)

            logger.info(f"▶ {v} | gemini")
            refined = refine_with_gemini(captions)

            out_file.write_text(
                f"VIDEO: {title}\n"
                f"URL: https://www.youtube.com/watch?v={v}\n\n"
                f"{refined}",
                encoding="utf-8",
            )

            outputs.append(out_file)
            state[v] = True
            save_state(state)

            logger.info(f"✓ Saved: {out_file.name}")

        except Exception as e:
            logger.error(f"✗ {v} failed: {e}")

    send_email(outputs, url)
    logger.info(f"Run complete in {time.time() - start:.2f}s")

# =========================================================
# ENTRY
# =========================================================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python youtube_hindi_captions_refine.py <youtube-url>")
        sys.exit(1)

    main(sys.argv[1])
