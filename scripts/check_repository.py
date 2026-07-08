#!/usr/bin/env python3
"""
Post-execution validation script.
Enforces publication boundaries and prevents sensitive data leaks.
"""
import os
import re
import sys
from pathlib import Path

# Secrets pattern (API keys, AWS tokens, generic secrets)
BANNED_PATTERNS = [
    re.compile(r"(?i)api[_-]?key[\s:=]+[\"'][a-zA-Z0-9_\-]{20,}"),
    re.compile(r"(?i)secret[\s:=]+[\"'][a-zA-Z0-9_\-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"C:\\Users\\[a-zA-Z0-9_]+", re.IGNORECASE),
]

# Banned file extensions
BANNED_EXTENSIONS = {".env", ".sqlite3", ".db", ".log", ".pem", ".key"}

def check_repo(root_dir: Path) -> int:
    errors = 0
    for filepath in root_dir.rglob("*"):
        if ".git" in str(filepath) or ".venv" in str(filepath) or "node_modules" in str(filepath):
            continue

        if filepath.is_file():
            if filepath.suffix in BANNED_EXTENSIONS:
                print(f"ERROR: Banned file extension found: {filepath.relative_to(root_dir)}")
                errors += 1
                continue

            # Only scan readable text files
            try:
                content = filepath.read_text(encoding="utf-8")
                for pattern in BANNED_PATTERNS:
                    match = pattern.search(content)
                    if match:
                        print(f"ERROR: Banned pattern found in {filepath.relative_to(root_dir)}")
                        errors += 1
            except UnicodeDecodeError:
                pass # Skip binaries

    if errors > 0:
        print(f"\nFailed repository check. {errors} violation(s) found.")
    else:
        print("Repository check passed. Ready for publication.")
    return errors

if __name__ == "__main__":
    sys.exit(check_repo(Path.cwd()))
