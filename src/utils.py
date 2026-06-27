import os
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector


def validate_required_envs():
  for k in ("OPENAI_API_KEY", "DATABASE_URL","PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")


def build_store(model: str) -> PGVector:
    """Get pgvector store configured"""
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")
    connection = os.getenv("DATABASE_URL")
    embeddings = OpenAIEmbeddings(
        model=model,
        openai_api_base="https://generative-ai-platform-development.ifoodcorp.com.br/api/v2",
    )

    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True
    )