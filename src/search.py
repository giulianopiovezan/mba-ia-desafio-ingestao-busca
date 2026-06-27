import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from utils import build_store, validate_required_envs

load_dotenv()
validate_required_envs()

PROMPT_TEMPLATE = """
CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_db(question: str, k: int) -> str:
  model = os.getenv("OPENAI_MODEL","text-embedding-3-small")
  store = build_store(model)

  rag_response = store.similarity_search_with_score(question, k)

  docs = []

  for doc, _ in rag_response:
    docs.append(doc.page_content.strip())
  
  return {
    "context": "\n\n".join(docs)
  }


def search_prompt(question=None):
  context = search_db(question, 10)

  prompt = ChatPromptTemplate.from_template(
    PROMPT_TEMPLATE,
  ).partial(
    context=context,
    question=question
  )

  return prompt
