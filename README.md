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
npm run dev
```

Creates the virtualenv, installs Python deps, copies `.env` if needed, then starts the agents API.

API docs: http://localhost:8080/docs
