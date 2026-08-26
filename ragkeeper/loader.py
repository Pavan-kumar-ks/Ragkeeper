from pathlib import Path


def discover_doc_files(repo_path: str, docs_root_subdir: str = "docs") -> list[Path]:
    docs_dir = Path(repo_path) / docs_root_subdir
    files = set(docs_dir.rglob("*.md")) | set(docs_dir.rglob("*.mdx"))
    return sorted(files)


def read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")
