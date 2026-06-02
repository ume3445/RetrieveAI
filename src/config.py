import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
CHROMA_PERSIST_DIR: str = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL: str = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL: str = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "64"))
TOP_K: int = int(os.environ.get("TOP_K", "5"))
