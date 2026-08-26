from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]


def split_markdown(raw_text: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False)
    header_docs = header_splitter.split_text(raw_text)

    size_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks: list[Document] = []
    for doc in header_docs:
        if len(doc.page_content) <= chunk_size:
            chunks.append(doc)
        else:
            chunks.extend(size_splitter.split_documents([doc]))
    return chunks


def header_hierarchy(chunk_metadata: dict) -> list[str]:
    return [chunk_metadata[key] for key in ("h1", "h2", "h3", "h4") if key in chunk_metadata]
