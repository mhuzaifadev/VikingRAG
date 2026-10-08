"""CLI entry: ``vikingrag-eval``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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

    from vikingrag.evaluation.runner import (
        DEFAULT_EVAL_METHODS,
        ManifestValidationError,
        load_examples_from_manifest,
        resolve_methods,
        run_evaluation,
    )

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
    try:
        examples = load_examples_from_manifest(path)
    except ManifestValidationError as exc:
        print(f"manifest_invalid: {exc}", file=sys.stderr)
        return 2
    if not examples:
        print("manifest has no usable examples", file=sys.stderr)
        return 1

    method_names = [
        p.strip() for p in (args.method or ",".join(DEFAULT_EVAL_METHODS)).split(",") if p.strip()
    ]
    echo_only = all(n.lower() in {"echo", "echo_placeholder"} for n in method_names)

    async def _run() -> Any:
        from vikingrag.evaluation.runner import EvalRunReport

        client = None
        try:
            if not echo_only:
                from vikingrag.client import VikingRAGClient

                client = VikingRAGClient.from_settings()
            methods = resolve_methods(method_names, client=client)
            result: EvalRunReport = await run_evaluation(
                examples=examples,
                methods=methods,
                judge=args.judge,
            )
            return result
        finally:
            if client is not None:
                await client.aclose()

    try:
        report_obj: Any = asyncio.run(_run())
    except Exception as exc:
        print(f"eval_run_failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    out = Path(args.output) if args.output else None
    payload = json.dumps(report_obj.to_dict(), indent=2) + "\n"
    scores = report_obj.measured_scores
    status = str(report_obj.status)
    if out:
        out.write_text(payload, encoding="utf-8")
        print(
            f"wrote={out} status={status} measured_scores={'set' if scores is not None else 'null'}"
        )
        if args.csv:
            from vikingrag.evaluation.runner import write_export_csv

            csv_path = Path(args.csv)
            write_export_csv(report_obj, csv_path)
            print(f"wrote_csv={csv_path}")
    else:
        print(payload, end="")
    if status in {"failed", "partial"}:
        return 1
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    import asyncio
    import json

    from vikingrag.client import VikingRAGClient

    async def _run() -> dict[str, object]:
        async with VikingRAGClient.from_settings() as client:
            return await client.explain(args.query_id)

    try:
        payload = asyncio.run(_run())
    except Exception as exc:
        print(f"explain_failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote={args.output}")
    else:
        print(text, end="")
    return 0


def _cmd_warmup(args: argparse.Namespace) -> int:
    import asyncio

    from vikingrag.domain.models.answer import ExecutionMode
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
    mode_name = (args.execution_mode or "vikingrag_e_plus").strip().lower()
    try:
        execution_mode = ExecutionMode(mode_name)
    except ValueError:
        print(
            f"invalid --execution-mode {args.execution_mode!r}; "
            "use vikingrag_e or vikingrag_e_plus",
            file=sys.stderr,
        )
        return 2
    print(
        f"warmup_plan dataset={plan.dataset} m={plan.m} "
        f"corpus_present={present} llm_configured={llm_ok} "
        f"materialize={not args.skip_materialize} mode={execution_mode.value}"
    )
    print(plan.notes)
    llm = None
    if llm_ok:
        try:
            llm = build_llm_provider(settings)
        except Exception as exc:
            print(f"llm_build_failed: {exc}", file=sys.stderr)
            llm_ok = False

    client = None
    if llm_ok and not args.skip_materialize:
        try:
            from vikingrag.client import VikingRAGClient

            client = VikingRAGClient.from_settings(settings)
        except Exception as exc:
            print(f"client_build_failed: {exc}", file=sys.stderr)

    try:
        result = run_warmup(
            plan,
            llm_configured=llm_ok,
            corpus_present=present,
            llm=llm,
            model=settings.llm.model or None,
            client=client,
            materialize=not args.skip_materialize,
            execution_mode=execution_mode,
            smoke_max=args.smoke_max,
        )
    except WarmupBlockedError as exc:
        print(exc.message, file=sys.stderr)
        return 2
    finally:
        if client is not None:
            asyncio.run(client.aclose())

    print(
        f"warmup_completed m_generated={result.m_generated} "
        f"jobs_drained={result.jobs_drained} edges_built={result.edges_built} "
        f"manifest={result.manifest_path}"
    )
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
    run.add_argument(
        "--method",
        type=str,
        default=None,
        help=(
            "Comma list of methods (default: vikingrag,vikingrag_e,vikingrag_e_plus). "
            "Use 'echo' for wiring-only placeholder."
        ),
    )
    run.add_argument(
        "--judge",
        type=str,
        default=None,
        help="Optional: 'scripted' for deterministic citation/non-empty rates (not LLM accuracy)",
    )
    run.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to write a flat CSV export alongside --output JSON",
    )
    run.set_defaults(func=_cmd_run)

    explain = sub.add_parser(
        "explain",
        help="Offline replay of a persisted query run (no new LLM calls)",
    )
    explain.add_argument(
        "--query-id", required=True, help="experience_query_run_id or answer query_id"
    )
    explain.add_argument("--output", type=str, default=None, help="Write JSON path")
    explain.set_defaults(func=_cmd_explain)

    warmup = sub.add_parser(
        "warmup",
        help="Plan/run historical-question warm-up (M=1000 default; blocked without data/LLM)",
    )
    warmup.add_argument("--dataset", required=True)
    warmup.add_argument("--m", type=int, default=1000, help="Number of historical questions")
    warmup.add_argument("--data-dir", default="data/eval")
    warmup.add_argument(
        "--execution-mode",
        default="vikingrag_e_plus",
        help="Mode used to materialize edges (vikingrag_e or vikingrag_e_plus)",
    )
    warmup.add_argument(
        "--skip-materialize",
        action="store_true",
        help="Generate questions only; do not run E+/drain edge builder",
    )
    warmup.add_argument(
        "--smoke-max",
        type=int,
        default=None,
        help="Cap generated questions for smoke runs (still labeled by --m in filename)",
    )
    warmup.set_defaults(func=_cmd_warmup)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
