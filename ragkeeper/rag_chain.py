import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from .config import get_settings
from .eval.pricing import estimate_cost
from .retrieval import BM25Index, HybridRetriever, Reranker, RetrievalPipeline, make_llm_query_expander
from .vectorstore import get_client, get_embeddings, get_vectorstore

FALLBACK_ANSWER = (
    "I don't have reliable information in the LangChain docs to answer that confidently."
)

SYSTEM_PROMPT = """You are RAGKeeper, an assistant that answers questions about LangChain using \
only the provided documentation excerpts.
Rules:
- Answer only using the given context. If the context does not contain the answer, say you \
don't know rather than guessing.
- Be concise and technically accurate.
- Do not fabricate file paths, function names, or APIs that are not present in the context."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


def format_context(docs) -> str:
    parts = [f"[Source: {doc.metadata.get('source_path')}]\n{doc.page_content}" for doc in docs]
    return "\n\n---\n\n".join(parts)


class RagKeeperChat:
    def __init__(self):
        settings = get_settings()
        client = get_client(settings.qdrant_url)
        embeddings = get_embeddings(settings.embedding_model)
        vectorstore = get_vectorstore(client, settings.qdrant_collection, embeddings)
        bm25_index = BM25Index(client, settings.qdrant_collection)
        hybrid_retriever = HybridRetriever(vectorstore, bm25_index, k=settings.top_k, candidate_k=20)
        reranker = Reranker()
        self.llm = ChatGroq(model=settings.groq_model, temperature=0, api_key=settings.groq_api_key)
        query_expander = make_llm_query_expander(self.llm) if settings.enable_query_expansion else None
        self.retriever = RetrievalPipeline(
            hybrid_retriever, reranker, k=settings.top_k, pool_size=20, query_expander=query_expander
        )
        self.confidence_threshold = settings.confidence_threshold
        self.model = settings.groq_model

    def answer(self, question: str, retrieval_query: str | None = None) -> dict:
        retrieval_start = time.perf_counter()
        trace = self.retriever.retrieve_with_trace(retrieval_query or question)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        if not trace or trace[0]["rerank_score"] < self.confidence_threshold:
            return {
                "answer": FALLBACK_ANSWER,
                "sources": [],
                "trace": trace,
                "retrieval_latency_ms": retrieval_latency_ms,
                "generation_latency_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
            }

        docs = [entry["doc"] for entry in trace]
        context = format_context(docs)

        generation_start = time.perf_counter()
        messages = PROMPT.format_messages(context=context, question=question)
        ai_message = self.llm.invoke(messages)
        generation_latency_ms = (time.perf_counter() - generation_start) * 1000

        token_usage = ai_message.response_metadata.get("token_usage", {})
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)

        sources: list[str] = []
        for doc in docs:
            src = doc.metadata.get("source_path")
            if src and src not in sources:
                sources.append(src)

        return {
            "answer": ai_message.content,
            "sources": sources,
            "trace": trace,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": estimate_cost(self.model, prompt_tokens, completion_tokens),
        }
