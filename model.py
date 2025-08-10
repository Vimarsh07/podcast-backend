

from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, UniqueConstraint, Index, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from db import Base

from enum import Enum as PyEnum

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    subscriptions = relationship("UserPodcast", back_populates="user", cascade="all, delete")

class Podcast(Base):
    __tablename__ = "podcasts"
    id = Column(Integer, primary_key=True, index=True)
    feed_url = Column(String, unique=True, nullable=False)
    title = Column(String)
    last_fetched = Column(DateTime)

    episodes = relationship("Episode", back_populates="podcast", cascade="all, delete")
    subscribers = relationship("UserPodcast", back_populates="podcast", cascade="all, delete")

class UserPodcast(Base):
    __tablename__ = "user_podcasts"
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    podcast_id = Column(Integer, ForeignKey("podcasts.id", ondelete="CASCADE"), primary_key=True)
    subscribed_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="subscriptions")
    podcast = relationship("Podcast", back_populates="subscribers")



class TranscriptStatus(PyEnum):
    NOT_REQUESTED = "NOT_REQUESTED"
    QUEUED = "QUEUED"
    TRANSCRIBING = "TRANSCRIBING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Episode(Base):
    __tablename__ = "episodes"

    id = Column(Integer, primary_key=True, index=True)
    podcast_id = Column(Integer, ForeignKey("podcasts.id", ondelete="CASCADE"), nullable=False)

    # identity & metadata
    guid = Column(String, nullable=False)
    title = Column(String, nullable=True)
    pub_date = Column(DateTime, index=True)
    summary = Column(Text)

    # extras for richer cards in the UI
    duration_seconds = Column(Integer, nullable=True)   # parsed from itunes:duration
    image_url = Column(String, nullable=True)           # episode image if present

    # transcription pipeline
    transcript = Column(Text)
    transcript_status = Column(
        Enum(TranscriptStatus, name="transcript_status_enum"),
        nullable=False,
        default=TranscriptStatus.NOT_REQUESTED,
        server_default="NOT_REQUESTED",
    )

    # audio
    audio_url = Column(String, nullable=True)           # enclosure URL; download only when requested

    __table_args__ = (
        UniqueConstraint("podcast_id", "guid", name="uq_episode"),
        Index("ix_episodes_podcast_pubdate", "podcast_id", "pub_date"),
    )

    podcast = relationship("Podcast", back_populates="episodes")
