from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Matchr AI Service"
    app_env: str = "development"
    api_prefix: str = "/api"
    cors_origins: str = "http://localhost:3000,http://localhost:4000"
    port: int = 8080

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    kong_gateway_url: str = ""
    embedding_model: str = "text-embedding-3-small"

    supabase_url: str = ""
    supabase_service_role_key: str = ""
    serpapi_api_key: str = ""
    nutrient_api_key: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
