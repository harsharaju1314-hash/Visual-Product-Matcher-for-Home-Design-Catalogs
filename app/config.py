from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings and environment variable configuration.
    Loads from system environment variables or a local .env file.
    """
    APP_NAME: str = "Visual Product Matcher for Home-Design Catalogs"
    APP_ENV: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # PostgreSQL & pgvector Database URL
    DATABASE_URL: str = "postgresql://cyncly_user:cyncly_password@localhost:5432/product_catalog_db"

    # Image validation limits
    MAX_UPLOAD_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 Megabytes
    ALLOWED_IMAGE_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "webp"]

    # Pretrained Vision Model Config
    MODEL_NAME: str = "resnet18"
    EMBEDDING_DIM: int = 512
    DEFAULT_TOP_K: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
