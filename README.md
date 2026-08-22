# Matchr AI Service

Agent and pipeline service for grounding, matching, tailoring, growth plans, event ranking, and outreach drafts. The Backend calls these endpoints; this service does not serve the product UI.

## Agents

| Agent | Role |
| :--- | :--- |
| A Grounding Auditor | LinkedIn + resume → Ground Truth Profile |
| C Semantic Match | pgvector + LLM Fit Scorecard |
| D Tailoring | Hallucination-free resume / cover letter |
| E Outreach Draft | Gmail-ready cold email (never auto-sends) |
| F Career Growth Advisor | Courses, YouTube, certs for target skills |
| G Event Match | Top compatible networking events |

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8001
```

API docs: http://localhost:8001/docs
