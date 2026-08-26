from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

EMBEDDING_DIM = 384  # sentence-transformers/all-MiniLM-L6-v2


def get_client(url: str) -> QdrantClient:
    return QdrantClient(url=url)


def get_embeddings(model: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model)


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int = EMBEDDING_DIM, recreate: bool = False) -> None:
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )


def delete_by_source_path(client: QdrantClient, collection_name: str, source_path: str) -> None:
    client.delete(
        collection_name=collection_name,
        points_selector=qmodels.FilterSelector(
            filter=qmodels.Filter(
                must=[qmodels.FieldCondition(key="metadata.source_path", match=qmodels.MatchValue(value=source_path))]
            )
        ),
    )


def get_vectorstore(client: QdrantClient, collection_name: str, embeddings: HuggingFaceEmbeddings) -> QdrantVectorStore:
    return QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)
