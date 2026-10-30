#!/usr/bin/env python
"""CLI базы нормативных документов: единственная точка входа для человека и агента.

Подкоманды:

    add      индексировать документ (PDF -> Markdown -> SQLite -> Qdrant)
    search   поиск: hybrid (по умолчанию) / semantic / keyword
    clause   точная ссылка на пункт: код документа + номер пункта
    docs     реестр проиндексированных документов
    status   сверка SQLite с Qdrant (sync_delta = 0 — индекс цел)
    figures  картинки из PDF: список / --extract (вырезать) / --auto (LLM)
    describe описание картинки текстом (режим агента, пока vision.model пуст)
    tables   структурированные таблицы: список, ячейки с числами (--show), --extract
    formulas формулы из PDF: кропы + описание (--set/--auto), находятся FTS/вектором
    reembed  доэмбеджить чанки, оставшиеся без вектора (возобновление обрыва)

Флаг ``--json`` печатает один JSON-документ в stdout (без служебных строк) —
им пользуется агент-проверяющий; прогресс в этом режиме глушится.
"""

from __future__ import annotations

import argparse
import contextlib
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


def _silenced(json_mode: bool):
    """--json: служебный вывод сторонних библиотек (RapidOCR, pymupdf4llm)
    уходит в stderr, чтобы stdout остался чистым JSON для агента."""
    return (
        contextlib.redirect_stdout(sys.stderr)
        if json_mode
        else contextlib.nullcontext()
    )


# --------------------------------------------------------------- подкоманды
def cmd_add(args: argparse.Namespace) -> int:
    from norms_base.pipeline import add_document

    if not args.url and not args.pdf:
        print("нужен --url или --pdf", file=sys.stderr)
        return 2
    with _silenced(args.json):
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
            f"{result.vectors} точек Qdrant, {result.tables} таблиц, "
            f"{result.formulas} формул за {result.seconds:.1f} с"
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


def _figure_payload(row) -> dict:
    return {
        "figure_id": row["figure_id"],
        "doc_code": row["doc_code"],
        "page": row["page"],
        "kind": row["kind"],
        "caption": row["caption"] or "",
        "path": row["path"],
        "description": row["description"],
        "described_by": row["described_by"],
        "described_at": row["described_at"],
        "pending": row["description"] is None,
    }


def cmd_figures(args: argparse.Namespace) -> int:
    from norms_base import figures as figures_mod
    from norms_base.storage import Store

    store = Store(DB_PATH)
    store.init_schema()
    payload = {"extracted": None, "described": None, "figures": []}
    try:
        if args.extract:
            docs = [store.get_document(args.doc)] if args.doc else store.list_documents()
            total = 0
            for doc in docs:
                if doc is None:
                    print(f"документ {args.doc} не найден", file=sys.stderr)
                    return 1
                if not doc["pdf_path"]:
                    if not args.json:
                        print(f"пропуск {doc['code']}: нет pdf_path", file=sys.stderr)
                    continue
                total += figures_mod.sync_figures(
                    store, doc["doc_id"], doc["pdf_path"], doc["code"],
                    verbose=not args.json,
                )
            payload["extracted"] = total

        if args.auto:
            cfg = figures_mod.load_vision_config()
            if not cfg.auto:
                print(
                    "vision.model не задан в figures.yaml — автоописание выключено;",
                    "описывайте вручную: norms.py describe <id> --text '…'",
                    file=sys.stderr,
                )
                return 2
            pending = store.list_figures(args.doc, pending=True)
            if args.limit:
                pending = pending[: int(args.limit)]
            done = 0
            for row in pending:
                try:
                    text = figures_mod.describe_with_llm(row["path"], row["caption"] or "", cfg)
                except Exception as exc:  # noqa: BLE001 — одна картинка не роняет пакет
                    print(f"[img] id={row['figure_id']}: {type(exc).__name__}: {exc}", file=sys.stderr)
                    continue
                store.set_figure_description(row["figure_id"], text, by=f"llm:{cfg.model}")
                done += 1
            payload["described"] = done

        rows = store.list_figures(args.doc, pending=True if args.pending else None)
        payload["figures"] = [_figure_payload(row) for row in rows]
    finally:
        store.close()

    if args.json:
        _dump(payload)
    elif not payload["figures"]:
        print("картинок нет — сначала: norms.py figures --extract [--doc '<код>']")
    else:
        for index, item in enumerate(payload["figures"], start=1):
            status = item["description"] or "— (ожидает описания)"
            print(
                f"[{index}] id={item['figure_id']} {item['doc_code']} стр. {item['page']}"
                f" {item['kind']} {item['caption'] or '(без подписи)'}"
            )
            print(f"     файл: {item['path']}")
            print(f"     описание: {status}")
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    from norms_base.storage import Store

    store = Store(DB_PATH)
    store.init_schema()
    try:
        if store.get_figure(args.figure_id) is None:
            print(f"картинка id={args.figure_id} не найдена", file=sys.stderr)
            return 1
        figure, chunk_id = store.set_figure_description(
            args.figure_id, args.text, by=args.by,
        )
    finally:
        store.close()
    payload = {
        "figure_id": figure["figure_id"],
        "doc_code": figure["doc_code"],
        "page": figure["page"],
        "chunk_id": chunk_id,
        "described_by": figure["described_by"],
    }
    if args.json:
        _dump(payload)
    else:
        target = f"чанк {chunk_id}" if chunk_id else "чанк не найден (описание сохранено в figures)"
        print(f"id={payload['figure_id']}: описание записано, вшито в {target}")
        print("вектор чанка устарел — norms.py reembed, потом status (СИНХРОНО)")
    return 0


def cmd_tables(args: argparse.Namespace) -> int:
    from norms_base import tables as tables_mod
    from norms_base.convert import pdf_to_markdown
    from norms_base.storage import Store

    store = Store(DB_PATH)
    store.init_schema()
    payload = {"extracted": None, "tables": []}
    try:
        if args.extract:
            docs = [store.get_document(args.doc)] if args.doc else store.list_documents()
            total = 0
            with _silenced(args.json):
                for doc in docs:
                    if doc is None:
                        print(f"документ {args.doc} не найден", file=sys.stderr)
                        return 1
                    if not doc["pdf_path"]:
                        if not args.json:
                            print(f"пропуск {doc['code']}: нет pdf_path", file=sys.stderr)
                        continue
                    md = pdf_to_markdown(doc["pdf_path"], doc["code"])
                    total += tables_mod.sync_tables(
                        store, doc["doc_id"], doc["code"], md.text, md.page_for,
                        verbose=not args.json,
                    )
            payload["extracted"] = total

        if args.show is not None:
            table = store.get_table(args.show)
            if table is None:
                print(f"таблица id={args.show} не найдена", file=sys.stderr)
                return 1
            payload["tables"] = [table]
        else:
            for row in store.list_tables(args.doc):
                payload["tables"].append(
                    {
                        "table_id": row["table_id"],
                        "doc_code": row["doc_code"],
                        "caption": row["caption"] or "",
                        "page_start": row["page_start"],
                        "page_end": row["page_end"],
                        "chunk_id": row["chunk_id"],
                        "row_count": row["row_count"],
                        "col_count": row["col_count"],
                    }
                )
    finally:
        store.close()

    if args.json:
        _dump(payload)
    elif args.show is not None:
        table = payload["tables"][0]
        pages = (
            f"стр. {table['page_start']}"
            if table["page_start"] == table["page_end"]
            else f"стр. {table['page_start']}-{table['page_end']}"
        )
        print(f"[{table['table_id']}] {table['doc_code']} {pages} {table['caption']}")
        print("шапка: " + " | ".join(table["header"]))
        for index, row in enumerate(table["rows"], start=1):
            cells = []
            for cell in row:
                cells.append(cell["raw"] if cell["num"] is None else f"{cell['raw']}={cell['num']:g}")
            print(f"  {index}: " + " | ".join(cells))
    elif not payload["tables"]:
        print("таблиц нет — сначала: norms.py tables --extract [--doc '<код>']")
    else:
        for item in payload["tables"]:
            cap = item["caption"] or "(без подписи)"
            pages = (
                f"стр. {item['page_start']}"
                if item["page_start"] == item["page_end"]
                else f"стр. {item['page_start']}-{item['page_end']}"
            )
            print(
                f"[{item['table_id']}] {item['doc_code']} {pages} "
                f"{item['row_count']}x{item['col_count']} {cap}"
            )
    return 0


def _formula_payload(row) -> dict:
    return {
        "formula_id": row["formula_id"],
        "doc_code": row["doc_code"],
        "tag": row["tag"],
        "page": row["page"],
        "path": row["path"],
        "anchor": row["anchor"] or "",
        "latex": row["latex"],
        "expr": row["expr"],
        "params": row["params"],
        "description": row["description"],
        "pending": not any([row["description"], row["latex"], row["expr"]]),
    }


def cmd_formulas(args: argparse.Namespace) -> int:
    from norms_base import formulas as formulas_mod
    from norms_base.figures import load_vision_config
    from norms_base.storage import Store

    store = Store(DB_PATH)
    store.init_schema()
    payload = {"extracted": None, "set": None, "auto": None, "dropped": None, "formulas": []}
    try:
        if args.extract:
            docs = [store.get_document(args.doc)] if args.doc else store.list_documents()
            total = 0
            with _silenced(args.json):
                for doc in docs:
                    if doc is None:
                        print(f"документ {args.doc} не найден", file=sys.stderr)
                        return 1
                    if not doc["pdf_path"]:
                        if not args.json:
                            print(f"пропуск {doc['code']}: нет pdf_path", file=sys.stderr)
                        continue
                    total += formulas_mod.sync_formulas(
                        store, doc["doc_id"], doc["pdf_path"], doc["code"],
                        verbose=not args.json,
                    )
            payload["extracted"] = total

        if args.set_id is not None:
            params = json.loads(args.params) if args.params else None
            row, chunk_id = store.set_formula_data(
                args.set_id, latex=args.latex, expr=args.expr,
                params=params, description=args.text, by=args.by,
            )
            payload["set"] = {
                "formula_id": row["formula_id"],
                "doc_code": row["doc_code"],
                "chunk_id": chunk_id,
            }

        if args.auto:
            cfg = load_vision_config()
            if not cfg.auto:
                print(
                    "vision.model не задан в figures.yaml — автоописание выключено;",
                    "описывайте вручную: norms.py formulas --set <id> --text '...'",
                    file=sys.stderr,
                )
                return 2
            pending = store.list_formulas(args.doc, pending=True)
            if args.limit:
                pending = pending[: int(args.limit)]
            done = 0
            for row in pending:
                try:
                    data = formulas_mod.describe_formula_with_llm(
                        row["path"], row["anchor"] or "", row["tag"], cfg,
                    )
                except Exception as exc:  # noqa: BLE001 — одна формула не роняет пакет
                    print(f"[fml] id={row['formula_id']}: {type(exc).__name__}: {exc}", file=sys.stderr)
                    continue
                store.set_formula_data(
                    row["formula_id"], latex=data["latex"], expr=data["expr"],
                    params=data["params"] or None, description=data["text"],
                    by=f"llm:{cfg.model}",
                )
                done += 1
            payload["auto"] = done

        if args.drop is not None:
            if store.get_formula(args.drop) is None:
                print(f"формула id={args.drop} не найдена", file=sys.stderr)
                return 1
            store.conn.execute("DELETE FROM formulas WHERE formula_id = ?", (args.drop,))
            store.commit()
            payload["dropped"] = args.drop

        rows = store.list_formulas(args.doc, pending=True if args.pending else None)
        payload["formulas"] = [_formula_payload(row) for row in rows]
    finally:
        store.close()

    if args.json:
        _dump(payload)
    elif not payload["formulas"]:
        print("формул нет — сначала: norms.py formulas --extract [--doc '<код>']")
    else:
        for item in payload["formulas"]:
            status = item["description"] or item["latex"] or "— (ожидает описания)"
            tag = f"({item['tag']})" if item["tag"] else "(без номера)"
            print(f"[{item['formula_id']}] {item['doc_code']} стр. {item['page']} {tag}")
            print(f"     файл: {item['path']}")
            if item["anchor"]:
                print(f"     рядом: {item['anchor'][:100]}")
            print(f"     описание: {status}")
        if payload["set"]:
            target = payload["set"]["chunk_id"]
            print(f"\nзаписано: id={payload['set']['formula_id']} в чанк {target or 'не найден'}")
            if target:
                print("вектор чанка устарел — norms.py reembed, потом status (СИНХРОНО)")
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

    figures_p = sub.add_parser(
        "figures", parents=[common], help="картинки из PDF: список/извлечение/автоописание"
    )
    figures_p.add_argument("--doc", help="только этот документ (код)")
    figures_p.add_argument("--pending", action="store_true", help="только ещё не описанные")
    figures_p.add_argument(
        "--extract", action="store_true",
        help="вырезать картинки из PDF (заново, идемпотентно)",
    )
    figures_p.add_argument(
        "--auto", action="store_true",
        help="описать pending внешней LLM (нужен vision.model в figures.yaml)",
    )
    figures_p.add_argument("--limit", type=int, help="сколько pending описать в --auto")
    figures_p.set_defaults(func=cmd_figures)

    describe_p = sub.add_parser(
        "describe", parents=[common], help="записать описание картинки (режим агента)",
    )
    describe_p.add_argument("figure_id", type=int, help="id из norms.py figures")
    describe_p.add_argument("--text", required=True, help="текст описания")
    describe_p.add_argument("--by", default="agent", help="кто описал (agent / имя модели)")
    describe_p.set_defaults(func=cmd_describe)

    docs = sub.add_parser("docs", parents=[common], help="реестр документов")
    docs.set_defaults(func=cmd_docs)

    stat = sub.add_parser("status", parents=[common], help="сверка SQLite <-> Qdrant")
    stat.set_defaults(func=cmd_status)

    reemb = sub.add_parser("reembed", parents=[common], help="доэмбеджить чанки без векторов")
    reemb.add_argument("--limit", type=int, help="ограничить количество чанков")
    reemb.set_defaults(func=cmd_reembed)

    tables_p = sub.add_parser(
        "tables", parents=[common], help="структурированные таблицы: список/ячейки/извлечение"
    )
    tables_p.add_argument("--doc", help="только этот документ (код)")
    tables_p.add_argument(
        "--extract", action="store_true",
        help="разобрать таблицы из Markdown заново (идемпотентно)",
    )
    tables_p.add_argument(
        "--show", type=int, metavar="ID",
        help="все ячейки таблицы с числами (id из списка)",
    )
    tables_p.set_defaults(func=cmd_tables)

    formulas_p = sub.add_parser(
        "formulas", parents=[common], help="формулы из PDF: список/извлечение/описание"
    )
    formulas_p.add_argument("--doc", help="только этот документ (код)")
    formulas_p.add_argument(
        "--extract", action="store_true",
        help="вырезать формулы из PDF заново (идемпотентно)",
    )
    formulas_p.add_argument("--pending", action="store_true", help="только ещё не описанные")
    formulas_p.add_argument(
        "--set", type=int, dest="set_id", metavar="ID",
        help="записать распознавание формулы (см. --text/--latex/--expr/--params)",
    )
    formulas_p.add_argument("--text", help="прозаическое описание (для поиска)")
    formulas_p.add_argument("--latex", help="формула в LaTeX")
    formulas_p.add_argument("--expr", help="вычисляемое выражение (Python)")
    formulas_p.add_argument("--params", help="JSON: символ -> расшифровка с единицами")
    formulas_p.add_argument("--by", default="agent", help="кто описал (agent / имя модели)")
    formulas_p.add_argument(
        "--auto", action="store_true",
        help="описать pending внешней LLM (нужен vision.model в figures.yaml)",
    )
    formulas_p.add_argument("--limit", type=int, help="сколько pending описать в --auto")
    formulas_p.add_argument("--drop", type=int, metavar="ID", help="удалить ложную формулу")
    formulas_p.set_defaults(func=cmd_formulas)

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
