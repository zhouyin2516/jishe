from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # Application Config
    INTERNAL_API_KEY: str = Field(..., description="API key used to authenticate internal requests")
    
    # OBS Configuration
    OBS_BUCKET: str = Field(..., description="OBS Bucket name")
    OBS_ENDPOINT: str = Field(..., description="OBS Endpoint, e.g., obs.cn-north-4.myhuaweicloud.com")
    
    # IAM Region Configuration
    IAM_REGION: str = Field(..., description="IAM Region for STS generation, e.g., cn-north-4")
    
    # Authentication Mode Flow
    USE_ECS_AGENCY: bool = Field(
        default=True, 
        description="Set to True for cloud deployment (uses 169.254 metadata). Set to False for local development."
    )
    
    # Dev Credentials (Only required if USE_ECS_AGENCY=False)
    DEV_AK: str | None = Field(None, description="Static AK for local development")
    DEV_SK: str | None = Field(None, description="Static SK for local development")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


# Instantiate the global settings object
settings = Settings()
