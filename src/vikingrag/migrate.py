"""Run Alembic migrations from a pip-installed or editable checkout.

Usage::

    vikingrag-migrate upgrade head
    vikingrag-migrate current
    python -m vikingrag.migrate upgrade head
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


def _script_location() -> Path:
    """Prefer packaged migrations; fall back to repo-root ``migrations/``."""
    packaged = Path(__file__).resolve().parent / "migrations"
    if (packaged / "env.py").is_file():
        return packaged
    # Editable / git checkout: repo_root/migrations
    repo = Path(__file__).resolve().parents[2]
    root_migrations = repo / "migrations"
    if (root_migrations / "env.py").is_file():
        return root_migrations
    raise SystemExit(
        "Could not find Alembic migrations. Reinstall vikingrag or run from the git checkout."
    )


def _alembic_config() -> Any:
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(_script_location()))
    cfg.set_main_option("prepend_sys_path", ".")
    return cfg


def main(argv: list[str] | None = None) -> int:
    from alembic import command
    from alembic.config import Config

    parser = argparse.ArgumentParser(
        prog="vikingrag-migrate",
        description="Apply VikingRAG database migrations (uses VIKINGRAG_DATABASE_URL).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("upgrade", help="Upgrade to a revision (default: head)")
    up.add_argument("revision", nargs="?", default="head")

    cur = sub.add_parser("current", help="Show current revision")
    cur.add_argument("--verbose", "-v", action="store_true")

    hist = sub.add_parser("history", help="Show revision history")
    hist.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(argv)
    cfg = _alembic_config()
    assert isinstance(cfg, Config)

    if args.command == "upgrade":
        command.upgrade(cfg, args.revision)
        print(f"upgrade -> {args.revision}")
        return 0
    if args.command == "current":
        command.current(cfg, verbose=args.verbose)
        return 0
    if args.command == "history":
        command.history(cfg, verbose=args.verbose)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
