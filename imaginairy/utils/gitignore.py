import os
from pathlib import Path

import pathspec


def find_project_root(start_path):
    """
    Traverse up from a starting path to find the project root.

    The project root is identified by the presence of a '.git' directory inside it.
    """
    current_path = Path(start_path)

    while current_path != current_path.root:
        if (current_path / ".git").is_dir():
            return str(current_path)

        if (current_path / ".hg").is_dir():
            return current_path

        if (current_path / "pyproject.toml").is_file():
            return current_path

        if (current_path / "setup.py").is_file():
            return current_path

        current_path = current_path.parent

    return None


ALWAYS_IGNORE = """
.git
__pycache__
.direnv
.eggs
.git
.hg
.mypy_cache
.nox
.tox
.venv
venv
.svn
.ipynb_checkpoints
_build
buck-out
build
dist
__pypackages__
"""


def load_gitignore_spec_at_path(path):
    gitignore_path = os.path.join(path, ".gitignore")

    if os.path.exists(gitignore_path):
        with open(gitignore_path, encoding="utf-8") as f:
            patterns = f.read().split("\n")
            patterns.extend(ALWAYS_IGNORE.split("\n"))
        ignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    else:
        ignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", [])
    return ignore_spec


def get_nonignored_file_paths(directory, gitignore_dict=None, extensions=()):
    return_relative = False
    if gitignore_dict is None:
        gitignore_dict = {}
        return_relative = True
    gitignore_dict = {
        **gitignore_dict,
        directory: load_gitignore_spec_at_path(directory),
    }

    file_paths = []

    for entry in os.scandir(directory):
        if path_is_ignored(Path(entry.path), gitignore_dict):
            continue

        if entry.is_file():
            if any(entry.path.endswith(ext) for ext in extensions):
                continue

            file_paths.append(entry.path)

        elif entry.is_dir():
            subdir_file_paths = get_nonignored_file_paths(
                entry.path, gitignore_dict=gitignore_dict
            )
            file_paths.extend(subdir_file_paths)
    if return_relative:
        file_paths = [os.path.relpath(f, directory) for f in file_paths]
        file_paths.sort(key=lambda p: ("/" in p, p))

    return file_paths


def path_is_ignored(path: Path, gitignore_dict) -> bool:
    for gitignore_path, pattern in gitignore_dict.items():
        try:
            abspath = path if path.is_absolute() else Path.cwd() / path
            normalized_path = abspath.resolve()
            try:
                relative_path = normalized_path.relative_to(gitignore_path).as_posix()
            except ValueError:
                return False

        except OSError:
            return False

        if pattern.match_file(relative_path):
            return True
    return False
