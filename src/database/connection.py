import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


load_dotenv()


# Raw URL — used by psycopg / LangGraph checkpointer (keep this exported)
DATABASE_URL = os.getenv("DATABASE_URL")


if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in the environment.")


# SQLAlchemy URL — same string but with the +psycopg dialect marker
SQLALCHEMY_DATABASE_URL = DATABASE_URL.replace(
    "postgresql://",
    "postgresql+psycopg://",
    1
)


engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=300,
    pool_timeout=30,
    future=True
)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False
)