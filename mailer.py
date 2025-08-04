# mailer.py

import os
import logging
from email.mime.text import MIMEText
import smtplib

logger = logging.getLogger(__name__)

# Load these from your .env (or Render/Prod env) 
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
FROM_EMAIL = os.getenv("FROM_EMAIL")        # e.g. "noreply@yourdomain.com"
TO_EMAIL = os.getenv("NOTIFICATION_EMAIL")  # e.g. your user’s email

if not all([SMTP_HOST, SMTP_USER, SMTP_PASS, FROM_EMAIL, TO_EMAIL]):
    logger.warning("⚠️ Some mailer env vars are missing; emails will fail")

def send_new_episode_email(podcast_title: str, episode_title: str, episode_url: str):
    subject = f"New episode: {podcast_title} – {episode_title}"
    body = (
        f"A new episode has just been published on **{podcast_title}**:\n\n"
        f"Title: {episode_title}\n"
        f"Listen now: {episode_url}\n\n"
        "Enjoy!"
    )

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = TO_EMAIL

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(FROM_EMAIL, [TO_EMAIL], msg.as_string())
        logger.info("✅ Notification email sent")
    except Exception as e:
        logger.error(f"❌ Failed to send email: {e}")
        raise
