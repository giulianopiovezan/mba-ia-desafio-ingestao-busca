import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

for k in ("OPENAI_API_KEY", "DATABASE_URL","PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

PDF_PATH = os.getenv("PDF_PATH")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def load_pdf() -> list[Document]:
    """Load pdf documents to memory"""
    loader = PyPDFLoader(PDF_PATH)
    return loader.load()

def split_documents(docs: list[Document]) -> list[Document]: 
    """split loaded pdf docs using chunk size and overlap strategies"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=False
    )

    return splitter.split_documents(docs)

def enrich_documents(docs: list[Document]) -> list[Document]:
    """enrich documents with relavant metadatas"""
    return [
        Document(
            page_content=d.page_content,
            metadata= {
                k:v for k,v in d.metadata.items() if v not in ("", None)
            }
        )
        for d in docs
    ]

def build_store(model: str) -> PGVector:
    """Get pgvector store configured"""
    collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")
    connection = os.getenv("DATABASE_URL")
    embeddings = OpenAIEmbeddings(
        model=model,
    )

    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True
    )

def save_documents(
    ids: list[str],
    docs: list[Document]
):
    """save documents to database"""
    model = os.getenv("OPENAI_MODEL","text-embedding-3-small")
    store = build_store(model)

    store.add_documents(
        documents=docs,
        ids=ids
    )

def ingest_pdf():
    pdf_docs = load_pdf()
    splitted_documents = split_documents(pdf_docs)

    if not splitted_documents:
        raise "No splitted documents found."

    enriched_docs = enrich_documents(splitted_documents)
    
    ids = [f"doc-{i}" for i in range(len(enriched_docs))]
    
    save_documents(
        docs=enriched_docs,
        ids=ids
    )
    



if __name__ == "__main__":
    ingest_pdf()