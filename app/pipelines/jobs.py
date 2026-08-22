"""SerpApi Google Jobs harvest → embed → upsert to Supabase."""


def sync_jobs(query: str, location: str | None = None) -> dict:
    return {"synced": 0, "query": query, "location": location}
