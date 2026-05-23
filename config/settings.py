"""
Fashion Finder — Configuration
Loads all environment variables via Pydantic BaseSettings.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Pinecone
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "fashion-finder"

    # Paths
    DATA_DIR: str = "data/raw"
    PROCESSED_DIR: str = "data/processed"

    # Ingestion tuning
    BATCH_SIZE: int = 64
    SUBSET_SIZE: int = 10000

    @property
    def data_path(self) -> Path:
        return Path(self.DATA_DIR)

    @property
    def processed_path(self) -> Path:
        p = Path(self.PROCESSED_DIR)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def shape_pca_path(self) -> Path:
        return self.processed_path / "shape_pca.pkl"


# Singleton — import this everywhere
settings = Settings()
