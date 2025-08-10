# db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/podcastdb")

engine = create_engine(
    DATABASE_URL,
    echo=True,
    pool_pre_ping=True,     # ✅ revive dead connections
    pool_recycle=280,       # ✅ recycle before common 5‑min idle timeouts
    pool_size=5,
    max_overflow=10,
    connect_args={"sslmode": "require"} if "postgres" in DATABASE_URL else {},  # ✅ TLS on Render
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def init_db():
    from model import User, Podcast, Episode, UserPodcast
    Base.metadata.create_all(bind=engine)
