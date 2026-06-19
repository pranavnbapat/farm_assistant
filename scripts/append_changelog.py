#!/usr/bin/env python3
"""Append missing Git commits to CHANGELOG.md.

Default branch is master. Pass branch names as arguments to use a custom branch
set, for example:

    python3 scripts/append_changelog.py master release_1.0
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"
DEFAULT_BRANCHES = ["master"]
LISTED_HASH_PATTERN = re.compile(r"`([0-9a-f]{8})`", re.IGNORECASE)


@dataclass(frozen=True)
class Commit:
    hash: str
    date: str
    author: str
    subject: str
    branches: list[str]
    is_merge: bool

    @property
    def short_hash(self) -> str:
        return self.hash[:8]


def git(args: list[str]) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), *args],
        text=True,
    )


def existing_content() -> str:
    if CHANGELOG_PATH.exists():
        return CHANGELOG_PATH.read_text(encoding="utf-8")

    branches = " and ".join(f"`{branch}`" for branch in DEFAULT_BRANCHES)
    return "\n".join(
        [
            "# Farm Assistant Changelog",
            "",
            f"This changelog records Farm Assistant changes from the {branches} branch histories. Entries are grouped by commit author date and include the commit hash, branch history, merge marker where applicable, author, and commit subject.",
            "",
        ]
    )


def existing_hashes(content: str) -> set[str]:
    return {match.group(1).lower() for match in LISTED_HASH_PATTERN.finditer(content)}


def branch_commit_sets(branches: list[str]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}

    for branch in branches:
        commits = git(["rev-list", branch]).strip().splitlines()
        result[branch] = set(commits)

    return result


def collect_commits(branches: list[str], known_hashes: set[str]) -> list[Commit]:
    branch_sets = branch_commit_sets(branches)
    raw_log = git(
        [
            "log",
            "--date-order",
            "--date=short",
            "--format=%H%x1f%ad%x1f%an%x1f%s%x1f%P%x1e",
            *branches,
        ]
    )
    commits: list[Commit] = []

    for entry in raw_log.split("\x1e"):
        entry = entry.strip()

        if not entry:
            continue

        hash_, date, author, subject, parents = (entry.split("\x1f") + [""])[:5]
        short_hash = hash_[:8].lower()

        if short_hash in known_hashes:
            continue

        commit_branches = [branch for branch in branches if hash_ in branch_sets[branch]]
        is_merge = len(parents.split()) > 1
        commits.append(
            Commit(
                hash=hash_,
                date=date,
                author=author,
                subject=subject,
                branches=commit_branches,
                is_merge=is_merge,
            )
        )

    return commits


def format_commits(commits: list[Commit]) -> str:
    grouped: OrderedDict[str, list[Commit]] = OrderedDict()

    for commit in commits:
        grouped.setdefault(commit.date, []).append(commit)

    lines: list[str] = []

    for date, entries in grouped.items():
        lines.extend([f"## {date}", ""])

        for entry in entries:
            tags = [f"[{', '.join(entry.branches)}]"]

            if entry.is_merge:
                tags.append("[merge]")

            lines.append(
                f"- `{entry.short_hash}` {' '.join(tags)} - {entry.author} - {entry.subject}"
            )

        lines.append("")

    return "\n".join(lines).rstrip()


def append_content(content: str, addition: str) -> str:
    separator = "" if content.endswith("\n\n") else "\n" if content.endswith("\n") else "\n\n"
    return f"{content}{separator}{addition}\n"


def main() -> int:
    branches = sys.argv[1:] or DEFAULT_BRANCHES
    content = existing_content()
    commits = collect_commits(branches, existing_hashes(content))

    if not commits:
        print("No new commits to append.")
        return 0

    CHANGELOG_PATH.write_text(
        append_content(content, format_commits(commits)),
        encoding="utf-8",
    )
    suffix = "" if len(commits) == 1 else "s"
    print(f"Appended {len(commits)} commit{suffix} to CHANGELOG.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
