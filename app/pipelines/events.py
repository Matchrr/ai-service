"""SerpApi (or equivalent) event harvest → embed → upsert to Supabase."""


def sync_events(query: str, location: str | None = None) -> dict:
    return {"synced": 0, "query": query, "location": location}
