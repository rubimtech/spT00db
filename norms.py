#!/usr/bin/env python
"""CLI базы нормативных документов: единственная точка входа для человека и агента.

Подкоманды:

    add      индексировать документ (PDF -> Markdown -> SQLite -> Qdrant)
    search   поиск: hybrid (по умолчанию) / semantic / keyword
    clause   точная ссылка на пункт: код документа + номер пункта
    docs     реестр проиндексированных документов
    status   сверка SQLite с Qdrant (sync_delta = 0 — индекс цел)
    reembed  доэмбеджить чанки, оставшиеся без вектора (возобновление обрыва)

Флаг ``--json`` печатает один JSON-документ в stdout (без служебных строк) —
им пользуется агент-проверяющий; прогресс в этом режиме глушится.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from norms_base import search  # noqa: E402
from norms_base.config import DB_PATH  # noqa: E402

DEFAULT_LIMIT = 400  # символов текста чанка в выдаче; --full снимает обрезку


# ------------------------------------------------------------------ вывод
def _dump(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _hits_payload(hits: list[search.Hit], limit: int | None) -> list[dict]:
    return [hit.as_dict(text_limit=limit) for hit in hits]


def _print_hits(hits: list[search.Hit], limit: int | None) -> None:
    if not hits:
        print("ничего не найдено")
        return
    for rank, hit in enumerate(hits, start=1):
        sources = ",".join(hit.sources)
        print(f"[{rank}] score={hit.score:.4f} ({sources}) {hit.cite}")
        if hit.clause_id or hit.heading:
            print(f"     {hit.heading or ''}".rstrip())
        text = hit.text if limit is None else hit.text[:limit]
        if limit is not None and len(hit.text) > limit:
            text += " …"
        for line in text.splitlines():
            print(f"     {line}")
        print()


# --------------------------------------------------------------- подкоманды
def cmd_add(args: argparse.Namespace) -> int:
    from norms_base.pipeline import add_document

    if not args.url and not args.pdf:
        print("нужен --url или --pdf", file=sys.stderr)
        return 2
    result = add_document(
        args.code,
        args.title,
        url=args.url,
        pdf_path=args.pdf,
        batch_size=args.batch,
        verbose=not args.json,
    )
    payload = dataclasses.asdict(result)
    payload["md_path"] = str(result.md_path)
    payload["pdf_path"] = str(result.pdf_path)
    if args.json:
        _dump(payload)
    else:
        print(
            f"итог: {result.chunks} чанков, {result.embedded} векторов, "
            f"{result.vectors} точек Qdrant за {result.seconds:.1f} с"
        )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    limit = None if args.full else args.limit
    if args.mode == "hybrid":
        hits = search.hybrid(args.query, k=args.k, doc_code=args.doc)
    elif args.mode == "semantic":
        hits = search.semantic(
            args.query,
            k=args.k,
            doc_code=args.doc,
            score_threshold=args.threshold,
        )
    elif args.mode == "keyword":
        hits = search.keyword(args.query, k=args.k, doc_code=args.doc)
    else:  # pragma: no cover - argparse ограничивает choices
        raise ValueError(f"неизвестный режим: {args.mode}")
    if args.json:
        _dump(_hits_payload(hits, limit))
    else:
        _print_hits(hits, limit)
    return 0


def cmd_clause(args: argparse.Namespace) -> int:
    hits = search.clause(args.doc_code, args.clause_id)
    if args.json:
        _dump(_hits_payload(hits, None if args.full else args.limit))
    else:
        _print_hits(hits, None if args.full else args.limit)
    return 0


def cmd_docs(args: argparse.Namespace) -> int:
    rows = search.docs()
    if args.json:
        _dump(rows)
    elif not rows:
        print("база пуста: norms.py add --code ... --title ... --url ...")
    else:
        for row in rows:
            print(f"{row['code']} | {row['title']} | {row['pages']} стр. | {row['chunks']} чанков | {row['indexed_at']}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from norms_base.pipeline import status
    from norms_base.storage import Store

    store = Store(DB_PATH)
    store.init_schema()
    try:
        info = status(store)
    finally:
        store.close()
    if args.json:
        _dump(info)
    else:
        for key, value in info.items():
            print(f"{key}: {value}")
        delta = info.get("sync_delta")
        if isinstance(delta, int):
            print("СИНХРОНО" if delta == 0 else f"РАСХОЖДЕНИЕ {delta:+d} — norms.py reembed")
    return 0


def cmd_reembed(args: argparse.Namespace) -> int:
    from norms_base.pipeline import reembed_pending
    from norms_base.storage import Store

    store = Store(DB_PATH)
    store.init_schema()
    try:
        done = reembed_pending(store, limit=args.limit, verbose=not args.json)
    finally:
        store.close()
    if args.json:
        _dump({"embedded": done})
    else:
        print(f"доэмбеджено: {done}")
    return 0


# ------------------------------------------------------------------ парсер
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="JSON в stdout (для агента), прогресс глушится")

    parser = argparse.ArgumentParser(
        prog="norms.py",
        description="База нормативных документов: SQLite FTS5 + Qdrant (bge-m3).",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", parents=[common], help="проиндексировать документ")
    add.add_argument("--code", required=True, help="код документа, например 'СП 60.13330.2020'")
    add.add_argument("--title", required=True, help="полное название документа")
    source = add.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="URL PDF (антибот-обход через download.fetch_pdf)")
    source.add_argument("--pdf", help="локальный PDF вместо скачивания")
    add.add_argument("--batch", type=int, default=64, help="размер батча upsert в Qdrant")
    add.set_defaults(func=cmd_add)

    search_p = sub.add_parser("search", parents=[common], help="поиск по базе")
    search_p.add_argument("query", help="запрос свободным текстом")
    search_p.add_argument("--mode", choices=["hybrid", "semantic", "keyword"], default="hybrid")
    search_p.add_argument("-k", "--k", type=int, default=8, help="сколько результатов вернуть")
    search_p.add_argument("--doc", help="ограничить один документ (код)")
    search_p.add_argument("--threshold", type=float, help="порог score для --mode semantic")
    search_p.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="обрезка текста чанка, символов")
    search_p.add_argument("--full", action="store_true", help="полный текст чанка без обрезки")
    search_p.set_defaults(func=cmd_search)

    clause = sub.add_parser("clause", parents=[common], help="точный пункт: код + номер")
    clause.add_argument("doc_code")
    clause.add_argument("clause_id", help="номер пункта, например 7.6.9")
    clause.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    clause.add_argument("--full", action="store_true")
    clause.set_defaults(func=cmd_clause)

    docs = sub.add_parser("docs", parents=[common], help="реестр документов")
    docs.set_defaults(func=cmd_docs)

    stat = sub.add_parser("status", parents=[common], help="сверка SQLite <-> Qdrant")
    stat.set_defaults(func=cmd_status)

    reemb = sub.add_parser("reembed", parents=[common], help="доэмбеджить чанки без векторов")
    reemb.add_argument("--limit", type=int, help="ограничить количество чанков")
    reemb.set_defaults(func=cmd_reembed)

    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except BrokenPipeError:  # pragma: no cover
        return 0
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except Exception as exc:  # noqa: BLE001
        if getattr(args, "json", False):
            _dump({"error": f"{type(exc).__name__}: {exc}"})
        else:
            print(f"ошибка: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
