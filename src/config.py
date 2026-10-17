from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache

class StorageSettings(BaseSettings):
    endpoint_url: str = "http://localhost:9000"
    access_key: str = "admin"
    secret_key: str = "password"
    hot_bucket: str = "hot-tier-bucket"
    cold_bucket: str = "cold-tier-bucket"
    
class PricingSettings(BaseSettings):
    hot_storage_per_gb_month: float = 0.023
    cold_storage_per_gb_month: float = 0.004
    egress_per_gb: float = 0.09
    latency_penalty_fixed: float = 0.01
    min_cold_duration_days: int = 90

class MLSettings(BaseSettings):
    horizon_days: int = 14

class Settings(BaseSettings):
    storage: StorageSettings = StorageSettings()
    pricing: PricingSettings = PricingSettings()
    ml: MLSettings = MLSettings()
    api_key: str = Field("sk-cloud-predict-production-key-123", env="API_KEY")
    database_url: str = Field("sqlite:///cloud_predict.db", env="DATABASE_URL")
    
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__")

@lru_cache()
def get_settings():
    return Settings()
