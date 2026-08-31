"""Canonical skill taxonomy copied from Backend/app/data/skills.py for harvest."""

from __future__ import annotations

import re

SKILL_ALIASES: dict[str, list[str]] = {
    "Python": ["python", "py3"],
    "TypeScript": ["typescript", "ts"],
    "JavaScript": ["javascript", "js", "es6"],
    "Go": ["golang", "go"],
    "Rust": ["rust"],
    "Java": ["java"],
    "SQL": ["sql", "ansi sql"],
    "FastAPI": ["fastapi"],
    "Django": ["django"],
    "Flask": ["flask"],
    "Node.js": ["node.js", "nodejs", "node"],
    "React": ["react", "react.js", "reactjs"],
    "Next.js": ["next.js", "nextjs"],
    "GraphQL": ["graphql"],
    "gRPC": ["grpc"],
    "REST APIs": ["rest api", "rest apis", "restful", "rest"],
    "PostgreSQL": ["postgresql", "postgres", "psql"],
    "Redis": ["redis"],
    "MongoDB": ["mongodb", "mongo"],
    "Elasticsearch": ["elasticsearch", "opensearch"],
    "pgvector": ["pgvector"],
    "Kafka": ["kafka"],
    "RabbitMQ": ["rabbitmq"],
    "Spark": ["spark", "pyspark"],
    "Airflow": ["airflow"],
    "dbt": ["dbt"],
    "Snowflake": ["snowflake"],
    "Docker": ["docker", "containerization"],
    "Kubernetes": ["kubernetes", "k8s", "eks", "gke"],
    "Terraform": ["terraform", "opentofu"],
    "AWS": ["aws", "amazon web services", "ec2", "lambda", "s3"],
    "GCP": ["gcp", "google cloud"],
    "Azure": ["azure"],
    "CI/CD": ["ci/cd", "continuous integration", "continuous delivery", "github actions", "jenkins"],
    "Observability": ["observability", "datadog", "prometheus", "grafana", "opentelemetry"],
    "System Design": ["system design", "distributed systems", "scalability", "high availability"],
    "Microservices": ["microservices", "service oriented"],
    "Machine Learning": ["machine learning", "ml", "predictive model"],
    "Deep Learning": ["deep learning", "neural network"],
    "PyTorch": ["pytorch", "torch"],
    "TensorFlow": ["tensorflow", "keras"],
    "LLMs": ["llm", "llms", "large language model", "gpt", "prompt engineering"],
    "RAG": ["rag", "retrieval augmented generation", "retrieval-augmented"],
    "LangChain": ["langchain", "langgraph"],
    "Vector Databases": ["vector database", "vector search", "pinecone", "weaviate", "embeddings"],
    "MLOps": ["mlops", "model deployment", "model serving"],
    "Data Modeling": ["data modeling", "dimensional model", "star schema"],
    "ETL": ["etl", "elt", "data pipeline", "data pipelines"],
    "Pandas": ["pandas", "numpy"],
    "Tableau": ["tableau", "looker", "power bi"],
    "A/B Testing": ["a/b testing", "experimentation", "ab test"],
    "Product Analytics": ["product analytics", "amplitude", "mixpanel"],
    "Testing": ["pytest", "unit test", "unit tests", "integration test", "test coverage", "jest"],
    "Security": ["oauth", "authentication", "authorization", "soc 2", "encryption"],
    "Leadership": ["mentoring", "mentored", "tech lead", "led a team", "cross-functional"],
    "Agile": ["agile", "scrum", "sprint planning"],
}

_ALIAS_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    skill: [re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])") for alias in aliases]
    for skill, aliases in SKILL_ALIASES.items()
}


def extract_skills(text: str) -> list[str]:
    lowered = text.lower()
    found: list[str] = []
    for skill, patterns in _ALIAS_PATTERNS.items():
        if any(pattern.search(lowered) for pattern in patterns):
            found.append(skill)
    return found
