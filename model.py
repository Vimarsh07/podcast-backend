

from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.orm import relationship
from datetime import datetime
from db import Base

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

class Episode(Base):
    __tablename__ = "episodes"
    id = Column(Integer, primary_key=True, index=True)
    podcast_id = Column(Integer, ForeignKey("podcasts.id", ondelete="CASCADE"), nullable=False)
    guid = Column(String, nullable=False)
    title = Column(String)
    pub_date = Column(DateTime)
    summary = Column(Text)
    transcript = Column(Text)
    audio_url = Column(String, nullable=True)    # stores the URL or path of downloaded audio

    __table_args__ = (
        UniqueConstraint("podcast_id", "guid", name="uq_episode"),
    )

    podcast = relationship("Podcast", back_populates="episodes")
