from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from .config import get_settings
from .vectorstore import get_client, get_embeddings, get_vectorstore

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
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
        self.llm = ChatGroq(model=settings.groq_model, temperature=0, api_key=settings.groq_api_key)
        self.chain = PROMPT | self.llm | StrOutputParser()

    def answer(self, question: str) -> tuple[str, list[str]]:
        docs = self.retriever.invoke(question)
        context = format_context(docs)
        answer_text = self.chain.invoke({"context": context, "question": question})

        sources: list[str] = []
        for doc in docs:
            src = doc.metadata.get("source_path")
            if src and src not in sources:
                sources.append(src)

        return answer_text, sources
