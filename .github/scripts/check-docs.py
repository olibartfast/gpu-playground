#!/usr/bin/env python3
"""Check docs/ house rules: names, header blocks, and index coverage.

Run from the repository root (CI does the same):

    python3 .github/scripts/check-docs.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = Path("docs")

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*\.md$")
ALLOWED_NAMES = {"README.md"}
HEADER_RE = re.compile(
    r"^> Audience: (?P<audience>.+?) · Applies to: (?P<applies>.+?) "
    r"· Last reviewed: (?P<reviewed>\d{4}-\d{2})$"
)
ALLOWED_AUDIENCE = {"humans", "agents"}
ALLOWED_APPLIES = {"all", "CUDA", "OpenCL", "OpenGL", "Vulkan"}


def tracked_docs() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "docs"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return [Path(p) for p in result.stdout.split("\0") if p.endswith(".md")]


def parse_header(text: str) -> dict | None:
    for line in text.splitlines()[:10]:
        match = HEADER_RE.match(line)
        if match:
            return match.groupdict()
    return None


def main() -> int:
    errors: list[str] = []
    files = sorted(tracked_docs())

    for path in files:
        if path.name not in ALLOWED_NAMES and not NAME_RE.match(path.name):
            errors.append(f"{path}: file name is not lowercase kebab-case")

    for directory in {str(path.parent) for path in files}:
        if directory == str(DOCS):
            continue
        for part in Path(directory).relative_to(DOCS).parts:
            if not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", part):
                errors.append(f"{directory}: directory name {part!r} is not lowercase kebab-case")

    for path in files:
        header = parse_header((ROOT / path).read_text(encoding="utf-8"))
        if header is None:
            errors.append(
                f"{path}: missing header block "
                "(expected '> Audience: ... · Applies to: ... · Last reviewed: YYYY-MM' "
                "within the first 10 lines)"
            )
            continue
        audience = {value.strip() for value in header["audience"].split(",")}
        applies = {value.strip() for value in header["applies"].split(",")}
        if not audience <= ALLOWED_AUDIENCE:
            errors.append(
                f"{path}: Audience must be a comma-separated subset of "
                f"{sorted(ALLOWED_AUDIENCE)}; got {sorted(audience)}"
            )
        if not applies <= ALLOWED_APPLIES:
            errors.append(
                f"{path}: Applies to must be a comma-separated subset of "
                f"{sorted(ALLOWED_APPLIES)}; got {sorted(applies)}"
            )
        month = int(header["reviewed"].split("-")[1])
        if not 1 <= month <= 12:
            errors.append(f"{path}: Last reviewed month {month} is invalid")

    index = (ROOT / DOCS / "README.md").read_text(encoding="utf-8")
    targets: set[str] = set()
    for match in re.finditer(r"\]\(([^)]+)\)", index):
        target = match.group(1).split("#", 1)[0].strip("<>")
        if not target or "://" in target:
            continue
        resolved = (ROOT / DOCS / target).resolve()
        try:
            targets.add(str(resolved.relative_to(ROOT.resolve())))
        except ValueError:
            continue

    for path in files:
        if path == DOCS / "README.md":
            continue
        if str(path) not in targets:
            errors.append(f"{path}: not linked from docs/README.md")

    if errors:
        print("docs checks failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"docs checks passed ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
