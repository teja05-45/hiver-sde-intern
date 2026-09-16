#!/usr/bin/env python3
"""
git filter-branch --msg-filter helper: strips AI-generated attribution from
commit messages while preserving the meaningful engineering content verbatim.

Removes exactly:
  - any line containing "Generated with Codebuff"
  - any line containing "Co-Authored-By: Codebuff"
  - the blank-line runs those leave behind at the end of the message

Usage (from repo root):
    git filter-branch -f --msg-filter 'python scripts/strip_commit_attribution.py' -- --all

Reads the original message on stdin, writes the cleaned message to stdout.
Never touches commit content, authors, or dates.
"""
import sys

FORBIDDEN_MARKERS = ("Generated with Codebuff", "Co-Authored-By: Codebuff")


def clean(message: str) -> str:
    lines = [ln for ln in message.splitlines()
             if not any(marker in ln for marker in FORBIDDEN_MARKERS)]
    # Drop blank lines that only existed to separate the removed trailer.
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines) + ("\n" if lines else "")


if __name__ == "__main__":
    sys.stdout.write(clean(sys.stdin.read()))
