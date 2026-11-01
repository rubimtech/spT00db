"""Готовит пакеты формул для суб-агентов и импортирует их ответы.

    python tools/formula_work.py split --parts 6
    python tools/formula_work.py import

Разделение на пакеты нужно, чтобы суб-агенты не писали в SQLite параллельно:
каждый кладёт ответ в ``data/formula_work/part_N.out.json``, а импорт идёт
одним процессом.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "formula_work"
DB = ROOT / "data" / "normbase.db"


def pending() -> list[dict]:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT f.formula_id, f.tag, f.page, f.path, f.anchor, f.latex_geom,"
        "       f.bbox, d.code AS doc_code"
        "  FROM formulas f JOIN documents d USING (doc_id)"
        " WHERE COALESCE(f.latex, '') = '' AND COALESCE(f.expr, '') = ''"
        "   AND COALESCE(f.description, '') = ''"
        " ORDER BY f.formula_id"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def split(parts: int) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    items = pending()
    if not items:
        print("нет формул без описания")
        return
    size = (len(items) + parts - 1) // parts
    for index in range(parts):
        chunk = items[index * size : (index + 1) * size]
        if not chunk:
            continue
        path = WORK / f"part_{index + 1}.json"
        path.write_text(
            json.dumps(chunk, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"{path.name}: {len(chunk)} формул (id {chunk[0]['formula_id']}…{chunk[-1]['formula_id']})")


def _norm(data: dict) -> dict:
    """Приводит ответ агента к полям set_formula_data (пустые -> None)."""
    params = data.get("params")
    if isinstance(params, str):
        text = params.strip()
        try:
            params = json.loads(text) if text else None
        except json.JSONDecodeError:
            params = None
    if not isinstance(params, dict) or not params:
        params = None
    return {
        "formula_id": int(data["formula_id"]),
        "latex": (data.get("latex") or "").strip() or None,
        "expr": (data.get("expr") or "").strip() or None,
        "params": params,
        "description": (data.get("text") or data.get("description") or "").strip() or None,
        "drop": bool(data.get("drop")),
        "by": str(data.get("by") or "agent:subagent"),
    }


def import_answers() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from norms_base.storage import Store

    store = Store(DB)
    store.init_schema()
    files = sorted(WORK.glob("part_*.out.json"))
    if not files:
        print("нет файлов part_*.out.json")
        return 1
    applied = dropped = skipped = 0
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"{path.name}: битый JSON — {exc}", file=sys.stderr)
            continue
        if isinstance(payload, dict):
            payload = payload.get("formulas") or payload.get("items") or []
        for raw in payload:
            if "formula_id" not in raw:
                continue
            item = _norm(raw)
            if item["drop"]:
                store.conn.execute(
                    "DELETE FROM formulas WHERE formula_id = ?", (item["formula_id"],)
                )
                dropped += 1
                continue
            if not (item["latex"] or item["expr"] or item["description"]):
                skipped += 1
                continue
            store.set_formula_data(
                item["formula_id"],
                latex=item["latex"],
                expr=item["expr"],
                params=item["params"],
                description=item["description"],
                by=item["by"],
            )
            applied += 1
    store.commit()
    store.close()
    print(f"записано {applied}, удалено ложных {dropped}, пропущено пустых {skipped}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    split_p = sub.add_parser("split")
    split_p.add_argument("--parts", type=int, default=6)
    sub.add_parser("import")
    args = parser.parse_args()
    if args.cmd == "split":
        split(args.parts)
        return 0
    return import_answers()


if __name__ == "__main__":
    raise SystemExit(main())
