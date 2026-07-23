from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Advanced AI Medical Intelligence Platform"
    environment: str = "development"
    database_url: str = f"sqlite:///{BASE_DIR / 'medical_ai.db'}"
    model_path: str = str(BASE_DIR / "artifacts" / "models" / "medical_cnn.pt")
    class_names: str = "Normal,Pneumonia,COVID-19,Tuberculosis"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    enable_llm: bool = False
    max_upload_mb: int = 8

    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")

    @property
    def classes(self) -> list[str]:
        return [item.strip() for item in self.class_names.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

