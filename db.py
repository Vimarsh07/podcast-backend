import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# load DATABASE_URL from .env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/podcastdb")

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def init_db():
    from model import User, Podcast, Episode, UserPodcast
    Base.metadata.create_all(bind=engine)
