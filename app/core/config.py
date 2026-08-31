from pydantic_settings import BaseSettings, SettingsConfigDict

_OPENAI_EMBEDDING_MODELS = {
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Matchr AI Service"
    app_env: str = "development"
    api_prefix: str = "/api"
    cors_origins: str = "http://localhost:3000,http://localhost:4000"
    port: int = 8080

    gemini_api_key: str = ""
    google_api_key: str = ""
    generative_language_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    kong_gateway_url: str = ""
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 1536

    supabase_url: str = ""
    supabase_service_role_key: str = ""
    serpapi_api_key: str = ""
    nutrient_api_key: str = ""

    xano_api_url: str = ""
    xano_api_key: str = ""
    xano_jobs_table: str = "matchrr_job_position"
    xano_search_run_table: str = "matchrr_job_search_run"
    xano_chunks_table: str = "matchrr_profile_chunk"
    xano_cover_letters_table: str = "matchrr_cover_letter"
    xano_match_endpoint: str = "match_jobs"  # Xano API endpoint path that runs the vector search function
    xano_chunk_match_function: str = ""  # optional; brute-force if empty

    harvest_standing_hours: int = 72
    harvest_standing_slots: int = 20
    harvest_cache_hours: int = 12
    harvest_stale_hours: int = 96
    harvest_pages: int = 2

    backend_url: str = "http://localhost:4000"
    matchr_service_secret: str = "matchr-dev"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def live_harvest_enabled(self) -> bool:
        return bool(self.serpapi_api_key and self.xano_api_url)

    @staticmethod
    def _normalize_gemini_key(key: str) -> str:
        key = key.strip()
        # New Google auth keys are AQ.... Pasting over an OpenAI sk- key often
        # leaves a leading "s" (sAQ....) which Gemini rejects as invalid.
        if key.startswith("sAQ."):
            return key[1:]
        return key

    @property
    def resolved_gemini_api_key(self) -> str:
        for key in (self.gemini_api_key, self.google_api_key, self.generative_language_api_key):
            if key.strip():
                return self._normalize_gemini_key(key)
        leftover = self.openai_api_key.strip()
        if leftover and not leftover.startswith("sk-"):
            return self._normalize_gemini_key(leftover)
        return ""

    @property
    def resolved_embedding_model(self) -> str:
        model = (self.embedding_model or "").strip()
        if not model or model in _OPENAI_EMBEDDING_MODELS:
            return "gemini-embedding-001"
        return model

    @property
    def embeddings_enabled(self) -> bool:
        return bool(self.resolved_gemini_api_key)


settings = Settings()
