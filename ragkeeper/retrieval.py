import re
from collections import defaultdict
from typing import Callable

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_RRF_K = 60
_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_EXPANSION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Generate {n} alternative phrasings of the user's question, useful for searching "
            "technical documentation. One per line, no numbering, no explanations.",
        ),
        ("human", "{question}"),
    ]
)


def make_llm_query_expander(llm, n: int = 2) -> Callable[[str], list[str]]:
    def expand(question: str) -> list[str]:
        try:
            response = llm.invoke(_EXPANSION_PROMPT.format_messages(question=question, n=n))
            lines = [line.strip("-*# ").strip() for line in response.content.splitlines()]
            return [line for line in lines if line][:n]
        except Exception:
            return []

    return expand


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _doc_key(doc: Document) -> str:
    return f"{doc.metadata.get('source_path')}::{doc.metadata.get('chunk_index')}"


class BM25Index:
    def __init__(self, client: QdrantClient, collection_name: str):
        self._docs: list[Document] = []
        points, next_offset = client.scroll(collection_name, limit=1000, with_payload=True, with_vectors=False)
        while True:
            for point in points:
                payload = point.payload or {}
                self._docs.append(
                    Document(page_content=payload.get("page_content", ""), metadata=payload.get("metadata", {}))
                )
            if next_offset is None:
                break
            points, next_offset = client.scroll(
                collection_name, limit=1000, offset=next_offset, with_payload=True, with_vectors=False
            )
        self._bm25 = BM25Okapi([_tokenize(doc.page_content) for doc in self._docs])

    def search(self, query: str, k: int) -> list[Document]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._docs[i] for i in ranked if scores[i] > 0]


class HybridRetriever:
    def __init__(
        self,
        vectorstore,
        bm25_index: BM25Index,
        k: int,
        candidate_k: int | None = None,
        dense_weight: float = 1.0,
        bm25_weight: float = 0.5,
    ):
        self.vectorstore = vectorstore
        self.bm25_index = bm25_index
        self.k = k
        self.candidate_k = candidate_k or max(k * 5, 20)
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

    def retrieve_candidates(self, question: str, n: int | None = None) -> list[Document]:
        dense_docs = self.vectorstore.similarity_search(question, k=self.candidate_k)
        bm25_docs = self.bm25_index.search(question, self.candidate_k)

        rrf_scores: dict[str, float] = defaultdict(float)
        doc_by_key: dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs):
            key = _doc_key(doc)
            rrf_scores[key] += self.dense_weight / (_RRF_K + rank + 1)
            doc_by_key[key] = doc

        for rank, doc in enumerate(bm25_docs):
            key = _doc_key(doc)
            rrf_scores[key] += self.bm25_weight / (_RRF_K + rank + 1)
            doc_by_key.setdefault(key, doc)

        ranked_keys = sorted(rrf_scores, key=lambda key: rrf_scores[key], reverse=True)
        return [doc_by_key[key] for key in ranked_keys[: n or self.k]]

    def retrieve(self, question: str) -> list[Document]:
        return self.retrieve_candidates(question, self.k)


class Reranker:
    def __init__(self, model_name: str = _RERANKER_MODEL):
        self._model = CrossEncoder(model_name)

    def rerank_with_scores(self, query: str, docs: list[Document], top_k: int) -> list[tuple[Document, float]]:
        if not docs:
            return []
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        return [(doc, float(score)) for doc, score in ranked[:top_k]]

    def rerank(self, query: str, docs: list[Document], top_k: int) -> list[Document]:
        return [doc for doc, _score in self.rerank_with_scores(query, docs, top_k)]


class RetrievalPipeline:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        k: int,
        pool_size: int | None = None,
        query_expander: Callable[[str], list[str]] | None = None,
    ):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.k = k
        self.pool_size = pool_size or max(k * 5, 20)
        self.query_expander = query_expander

    def _gather_candidates(self, question: str) -> list[Document]:
        queries = [question]
        if self.query_expander:
            queries += self.query_expander(question)

        rrf_scores: dict[str, float] = defaultdict(float)
        doc_by_key: dict[str, Document] = {}
        for query in queries:
            for rank, doc in enumerate(self.hybrid_retriever.retrieve_candidates(query, self.pool_size)):
                key = _doc_key(doc)
                rrf_scores[key] += 1 / (_RRF_K + rank + 1)
                doc_by_key.setdefault(key, doc)

        ranked_keys = sorted(rrf_scores, key=lambda key: rrf_scores[key], reverse=True)
        return [doc_by_key[key] for key in ranked_keys[: self.pool_size]]

    def retrieve(self, question: str) -> list[Document]:
        candidates = self._gather_candidates(question)
        return self.reranker.rerank(question, candidates, self.k)

    def retrieve_with_score(self, question: str) -> list[tuple[Document, float]]:
        candidates = self._gather_candidates(question)
        return self.reranker.rerank_with_scores(question, candidates, self.k)
