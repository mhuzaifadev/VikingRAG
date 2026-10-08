"""CLI entry: ``vikingrag-eval``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vikingrag.evaluation.adapters import ADAPTERS, get_adapter
from vikingrag.evaluation.base import DatasetNotAvailableError
from vikingrag.evaluation.smoke import report_json, run_offline_smoke


def _cmd_smoke(args: argparse.Namespace) -> int:
    if not args.offline:
        print(
            "Only --offline smoke is implemented. Full runs require datasets and providers.",
            file=sys.stderr,
        )
        return 2
    fixture = Path(args.fixture) if args.fixture else None
    try:
        report = run_offline_smoke(fixture=fixture)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(report_json(report), end="")
    return 0 if report.status == "ok" else 1


def _cmd_prepare(args: argparse.Namespace) -> int:
    adapter = get_adapter(args.dataset)
    base = Path(args.data_dir)
    if args.verify and not args.download:
        result = adapter.verify(base)
        print(f"dataset={adapter.manifest.name} ok={result.ok} message={result.message}")
        return 0 if result.ok else 1
    try:
        if args.download:
            path = adapter.download(base, force=args.force)
            print(f"downloaded={path}")
        if args.verify:
            result = adapter.verify(base)
            print(f"verify ok={result.ok} message={result.message}")
            return 0 if result.ok else 1
        adapter.require_present(base)
        print(f"present={adapter.data_root(base)}")
        return 0
    except DatasetNotAvailableError as exc:
        print(exc.message, file=sys.stderr)
        return 1


def _cmd_list(_args: argparse.Namespace) -> int:
    for name, adapter in sorted(ADAPTERS.items()):
        m = adapter.manifest
        print(f"{name}\t{m.version}\t{m.primary_source_url}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    import asyncio
    import json

    from vikingrag.evaluation.runner import load_examples_from_manifest, run_evaluation

    if not args.manifest:
        print(
            "vikingrag-eval run requires --manifest path/to/manifest.json "
            "with an examples/questions list. Status: BLOCKED.",
            file=sys.stderr,
        )
        return 2
    path = Path(args.manifest)
    if not path.is_file():
        print(f"manifest not found: {path}", file=sys.stderr)
        return 1
    examples = load_examples_from_manifest(path)
    if not examples:
        print("manifest has no usable examples", file=sys.stderr)
        return 1

    # Default: identity echo method (wiring). Wire VikingRAGClient methods in-process.
    async def _echo(question: str, *, document_ids: tuple[str, ...] = ()) -> dict[str, object]:
        del document_ids
        return {"answer": None, "citations": (), "usage": {}, "note": "echo_placeholder"}

    methods = {"echo_placeholder": _echo}
    if args.method:
        print(
            "Custom method wiring via CLI flags is not implemented; "
            "use the Python runner API with VikingRAGClient callables.",
            file=sys.stderr,
        )
        return 2

    report = asyncio.run(run_evaluation(examples=examples, methods=methods))
    out = Path(args.output) if args.output else None
    payload = json.dumps(report.to_dict(), indent=2) + "\n"
    if out:
        out.write_text(payload, encoding="utf-8")
        print(f"wrote={out} status={report.status} measured_scores=null")
    else:
        print(payload, end="")
    return 0


def _cmd_warmup(args: argparse.Namespace) -> int:
    from vikingrag.evaluation.adapters import get_adapter
    from vikingrag.evaluation.warmup import WarmupBlockedError, plan_warmup, run_warmup
    from vikingrag.providers.factory import build_llm_provider
    from vikingrag.settings.config import get_settings

    plan = plan_warmup(dataset=args.dataset, m=args.m, data_dir=args.data_dir)
    adapter = get_adapter(args.dataset)
    base = Path(args.data_dir)
    present = adapter.is_present(base)
    settings = get_settings()
    llm_ok = settings.llm.provider.lower().strip() not in {
        "unimplemented",
        "",
        "none",
        "fake",
        "test",
    } and bool(settings.llm.api_key)
    print(
        f"warmup_plan dataset={plan.dataset} m={plan.m} "
        f"corpus_present={present} llm_configured={llm_ok}"
    )
    print(plan.notes)
    llm = None
    if llm_ok:
        try:
            llm = build_llm_provider(settings)
        except Exception as exc:
            print(f"llm_build_failed: {exc}", file=sys.stderr)
            llm_ok = False
    try:
        result = run_warmup(
            plan,
            llm_configured=llm_ok,
            corpus_present=present,
            llm=llm,
            model=settings.llm.model or None,
        )
    except WarmupBlockedError as exc:
        print(exc.message, file=sys.stderr)
        return 2
    print(f"warmup_completed m_generated={result.m_generated} manifest={result.manifest_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vikingrag-eval",
        description="VikingRAG evaluation harness (never invents measured scores).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Deterministic offline scaffolding smoke")
    smoke.add_argument(
        "--offline",
        action="store_true",
        help="Required: run fixture-based offline smoke (no remote datasets)",
    )
    smoke.add_argument("--fixture", type=str, default=None, help="Override fixture markdown path")
    smoke.set_defaults(func=_cmd_smoke)

    prepare = sub.add_parser("prepare", help="Download/verify a dataset adapter")
    prepare.add_argument("--dataset", required=True, help="Dataset name (e.g. syllabusqa)")
    prepare.add_argument(
        "--data-dir",
        default="data/eval",
        help="Root directory for evaluation corpora",
    )
    prepare.add_argument("--download", action="store_true")
    prepare.add_argument("--verify", action="store_true")
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=_cmd_prepare)

    listing = sub.add_parser("list", help="List dataset adapters")
    listing.set_defaults(func=_cmd_list)

    run = sub.add_parser(
        "run",
        help="Run evaluation from a manifest (measured_scores null without a judge)",
    )
    run.add_argument("--manifest", type=str, default=None, help="JSON with examples/questions")
    run.add_argument("--output", type=str, default=None, help="Write report JSON path")
    run.add_argument("--method", type=str, default=None, help="Reserved for client wiring")
    run.set_defaults(func=_cmd_run)

    warmup = sub.add_parser(
        "warmup",
        help="Plan/run historical-question warm-up (M=1000 default; blocked without data/LLM)",
    )
    warmup.add_argument("--dataset", required=True)
    warmup.add_argument("--m", type=int, default=1000, help="Number of historical questions")
    warmup.add_argument("--data-dir", default="data/eval")
    warmup.set_defaults(func=_cmd_warmup)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
