from pathlib import Path

import git


def ensure_repo_synced(repo_url: str, repo_path: str) -> str:
    """Clone the repo if absent, otherwise pull. Returns the current commit hash."""
    path = Path(repo_path)

    if (path / ".git").exists():
        repo = git.Repo(path)
        repo.remotes.origin.pull()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        repo = git.Repo.clone_from(repo_url, path, depth=1)

    return repo.head.commit.hexsha
