# Contributing Guidelines

Thank you for your interest in contributing! This project is maintained as an open-source publication and research prototype.

## Publication Boundaries
Please adhere strictly to the following hygiene boundaries to keep this repository immaculate:
- **No Heavy Binaries:** Do not commit datasets, `.csv` caches, SQLite dumps, or model weights (`.pkl`, `.joblib`, etc.).
- **No Credentials:** Never commit `.env` files, API keys, or cloud access tokens. Use placeholders like `<redacted-api-key>` if documenting configuration files.
- **No Agent Logs:** If using AI assistants (e.g., Claude, Cursor), ensure folders like `.claude/` or `.cursor/` are excluded locally.

## Submission Process
1. Fork the repository and create your feature branch from `main`.
2. Ensure you have run the validation script locally: `python scripts/check_repository.py`.
3. Submit a Pull Request with a clear, professional description of the core value added.
