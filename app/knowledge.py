from functools import lru_cache

from groq import Groq
from langchain_openai import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
from app.config import settings
from sentence_transformers import SentenceTransformer


@lru_cache()
def clients():
    if not settings.groq_api_key or not settings.pinecone_api_key:
        raise ValueError("OpenAI and Pinecone API keys must be set in the environment variables.")
    return Groq(api_key=settings.groq_api_key), Pinecone(api_key=settings.pinecone_api_key)


@lru_cache()
def embedding_model():
    return SentenceTransformer(settings.hf_embedding_model)


def ensure_index():
    _, pc = clients()

    if not pc.has_index(settings.pinecone_index):
        pc.create_index(
            name=settings.pinecone_index,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    return pc.Index(settings.pinecone_index)


# def embed(texts):
#     ai, _ = clients()
#     return [x.embedding for x in ai.embeddings.create(model=settings.hf_embedding_model, input=texts).data]

def embed(texts):
    model = embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def search_kb(query, top_k=5):
    index = ensure_index()
    vector = embed([query])[0]
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [{'page': m.metadata.get('page'), 'text': m.metadata.get('text', ''), 'score': round(m.score, 3),
             'source': 'Gale Encyclopedia (2002)'} for m in result.matches if m.score >= 0.30]
