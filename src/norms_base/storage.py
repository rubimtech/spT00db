"""SQLite: документы, чанки и полнотекстовый индекс FTS5.

Зачем SQLite рядом с Qdrant:

* точные ссылки («п. 7.6.9», «приложение В») — это не семантика, а точное
  совпадение номера; искать их вектором бессмысленно и ненадёжно;
* FTS5 отдаёт лексический поиск (BM25) по терминам вроде «огнестойкость
  воздуховодов», который ловит формулировки, не совпадающие семантически;
* SQLite — источник правды: по нему считается статус индексации и
  восстановление Qdrant без повторной конверсии PDF.

Текст чанка хранится и в FTS, и в ``chunks``: это дубль (единицы МБ), но он
избавляет от триггеров external-content, которые ломаются при DELETE/INSERT
в пакетном режиме.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id        INTEGER PRIMARY KEY,
    code          TEXT NOT NULL UNIQUE,
    title         TEXT NOT NULL,
    url           TEXT,
    sha256        TEXT,
    source        TEXT,
    pdf_path      TEXT,
    md_path       TEXT,
    pages         INTEGER,
    chars         INTEGER,
    chunk_count   INTEGER,
    downloaded_at TEXT,
    converted_at  TEXT,
    indexed_at    TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      INTEGER PRIMARY KEY,
    doc_id        INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ord           INTEGER NOT NULL,
    kind          TEXT NOT NULL,
    clause_id     TEXT,
    section       TEXT,
    section_title TEXT,
    heading       TEXT,
    page_start    INTEGER,
    page_end      INTEGER,
    text          TEXT NOT NULL,
    embed_text    TEXT,
    embedded_at   TEXT,
    point_id      TEXT,
    UNIQUE (doc_id, ord)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_clause ON chunks(clause_id);

-- Все пункты, вошедшие в чанк (склейка мелких пунктов склеивает и номера).
-- Без этой таблицы «п. 7.6.9» не находился бы: пункт жил в чанке с primary
-- clause_id = 7.6.8, и точный поиск его не видел.
CREATE TABLE IF NOT EXISTS chunk_clauses (
    chunk_id   INTEGER NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    clause_id  TEXT NOT NULL,
    PRIMARY KEY (chunk_id, clause_id)
);
CREATE INDEX IF NOT EXISTS idx_chunk_clauses_clause ON chunk_clauses(clause_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, clause_id, section_title,
    chunk_id UNINDEXED, doc_code UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);

-- Описанные картинки PDF: файл лежит в data/img, description — источник
-- правды (переживает переиндексацию) и дублируется меткой <!-- fig:N -->
-- в тексте чанка, чтобы находиться FTS и вектором.
CREATE TABLE IF NOT EXISTS figures (
    figure_id    INTEGER PRIMARY KEY,
    doc_id       INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    page         INTEGER,
    kind         TEXT NOT NULL,          -- raster | crop
    xref         INTEGER,
    caption      TEXT,
    bbox         TEXT,
    path         TEXT NOT NULL,
    description  TEXT,
    described_by TEXT,
    described_at TEXT,
    UNIQUE (doc_id, path)
);
CREATE INDEX IF NOT EXISTS idx_figures_doc ON figures(doc_id);

-- Структурированные таблицы документа: ячейки с числами — источник правды
-- для расчётно-проверочных скриптов (текст чанка рядом, но скрипт берёт num).
CREATE TABLE IF NOT EXISTS tables (
    table_id    INTEGER PRIMARY KEY,
    doc_id      INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_id    INTEGER REFERENCES chunks(chunk_id) ON DELETE SET NULL,
    ord         INTEGER NOT NULL,
    caption     TEXT,
    page_start  INTEGER,
    page_end    INTEGER,
    header      TEXT NOT NULL,          -- JSON-массив шапки
    row_count   INTEGER NOT NULL,
    col_count   INTEGER NOT NULL,
    UNIQUE (doc_id, ord)
);
CREATE INDEX IF NOT EXISTS idx_tables_doc ON tables(doc_id);

-- Ячейки: row_idx 0 — шапка, 1..N строки данных; num — разобранное число
-- (NULL, если значение неоднозначно: «2/3», «–»), raw — источник правды.
CREATE TABLE IF NOT EXISTS table_cells (
    table_id INTEGER NOT NULL REFERENCES tables(table_id) ON DELETE CASCADE,
    row_idx  INTEGER NOT NULL,
    col_idx  INTEGER NOT NULL,
    raw      TEXT NOT NULL,
    num      REAL,
    PRIMARY KEY (table_id, row_idx, col_idx)
);

-- Формулы: кроп из PDF (data/img/<код>/f_pNNN_KK.png) + распознавание.
-- Описания переживают переиндексацию (path уникален в документе) и
-- вшиваются меткой <!-- fml:N --> в чанк — так находятся FTS и вектором.
CREATE TABLE IF NOT EXISTS formulas (
    formula_id   INTEGER PRIMARY KEY,
    doc_id       INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    tag          TEXT,                  -- номер формулы: «А.6» (или NULL)
    page         INTEGER,
    bbox         TEXT,
    path         TEXT NOT NULL,
    anchor       TEXT,                  -- якорный текст документа («…по формуле:»)
    latex        TEXT,
    latex_geom   TEXT,                  -- LaTeX из геометрии символов (черновик, без LLM)
    expr         TEXT,                  -- вычисляемое выражение для скриптов
    params       TEXT,                  -- JSON: символ -> расшифровка/единицы
    description  TEXT,                  -- проза для поиска (источник правды)
    described_by TEXT,
    described_at TEXT,
    UNIQUE (doc_id, path)
);
CREATE INDEX IF NOT EXISTS idx_formulas_doc ON formulas(doc_id);
"""

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _like_escape(value: str) -> str:
    r"""Экранирует спецсимволы LIKE (``%``, ``_``, ``\``) с ``ESCAPE '\'``.

    Коды документов пунктуацию содержат («СП 60.13330.2020»), но ``_`` и ``%``
    встречаются в кодах редко — экранируем всё, чтобы шаблон подмешивания
    правок не ловил лишние документы.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _unique(items: Iterable[Any]) -> list[str]:
    """Убирает дубли номеров пунктов, сохраняя порядок документа."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        value = str(item).strip()
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def parse_clause_ids(raw: Any) -> tuple[str, ...]:
    """``'7.6.8,7.6.9'`` -> ``('7.6.8', '7.6.9')``; пустая строка -> ``()``."""
    if not raw:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(str(item) for item in raw if str(item).strip())
    return tuple(part for part in str(raw).split(",") if part.strip())


#: Все пункты чанка в порядке документа (порядок обеспечивает ORDER BY rowid).
CLAUSE_IDS_EXPR = """COALESCE((SELECT group_concat(cl.clause_id, ',')
              FROM (SELECT clause_id FROM chunk_clauses WHERE chunk_id = c.chunk_id ORDER BY rowid) cl
             ), '')"""

def _strip_figure_note(text: str, figure_id: int) -> str:
    """Убирает старую метку описания картинки из текста (идемпотентная вшивка)."""
    return re.sub(
        rf"<!-- fig:{figure_id} -->.*?<!-- /fig:{figure_id} -->\n?",
        "", text, flags=re.S,
    )

def _strip_formula_note(text: str, formula_id: int) -> str:
    """Убирает старую метку описания формулы (идемпотентная вшивка)."""
    return re.sub(
        rf"<!-- fml:{formula_id} -->.*?<!-- /fml:{formula_id} -->\n?",
        "", text, flags=re.S,
    )


def _strip_all_formula_notes(text: str) -> str:
    """Убирает все метки описаний формул разом (перенос носителя, чистка)."""
    return re.sub(
        r"<!-- fml:\d+ -->.*?<!-- /fml:\d+ -->\n?", "", text, flags=re.S
    ).strip("\n")


def _formula_note(formula: sqlite3.Row) -> str | None:
    """Метка описания формулы для вшивки в чанк; None — пока описания нет.

    Номер («Формула (А.6)»), проза, выражение и параметры попадают в текст
    и в ``embed_text`` — так формула находится FTS и вектором вместе со
    своим пунктом, а скрипт берёт ``expr``/``params`` прямо из строки.
    """
    tag = (formula["tag"] or "").strip()
    label = f"Формула ({tag})" if tag else "Формула"
    description = (formula["description"] or "").strip()
    latex = (formula["latex"] or "").strip()
    expr = (formula["expr"] or "").strip()
    body = description or latex or expr
    if not body:
        return None
    lines = [f"**{label}** (стр. {formula['page']}): {body}"]
    if expr and expr not in body:
        lines.append(f"Выражение: {expr}")
    raw_params = formula["params"]
    if raw_params:
        try:
            params = json.loads(raw_params)
        except (TypeError, ValueError):
            params = {}
        if isinstance(params, dict) and params:
            lines.append(
                "Параметры: "
                + "; ".join(f"{key} — {value}" for key, value in params.items())
            )
    note_body = "\n".join(lines)
    return f"<!-- fml:{formula['formula_id']} -->\n{note_body}\n<!-- /fml:{formula['formula_id']} -->"


SELECT_COLUMNS = f"""
    c.chunk_id, c.doc_id, c.ord, c.kind, c.clause_id, c.section, c.section_title,
    c.heading, c.page_start, c.page_end, c.text,
    {CLAUSE_IDS_EXPR} AS clause_ids,
    d.code AS doc_code, d.title AS doc_title
"""


@dataclass(slots=True)
class ChunkRow:
    chunk_id: int
    doc_id: int
    ord: int
    kind: str
    clause_id: str | None
    section: str | None
    section_title: str | None
    heading: str | None
    page_start: int | None
    page_end: int | None
    text: str
    doc_code: str
    doc_title: str
    #: все пункты чанка в порядке документа (см. таблицу chunk_clauses)
    clause_ids: tuple[str, ...] = ()
    rank: float | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row, with_rank: bool = False) -> "ChunkRow":
        data = dict(row)
        primary = data.get("clause_id")
        raw_ids = data.get("clause_ids") or ""
        clause_ids = tuple(part for part in raw_ids.split(",") if part)
        if not clause_ids and primary:
            clause_ids = (primary,)
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            ord=data["ord"],
            kind=data["kind"],
            clause_id=data["clause_id"],
            section=data["section"],
            section_title=data["section_title"],
            heading=data["heading"],
            page_start=data["page_start"],
            page_end=data["page_end"],
            text=data["text"],
            doc_code=data["doc_code"],
            doc_title=data["doc_title"],
            clause_ids=clause_ids,
            rank=data.get("rank"),
        )


class Store:
    """Ленивое подключение: импорт модуля не открывает файл базы."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            self._conn = conn
        return self._conn

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    #: Колонки, добавленные после первой версии схемы: ``ALTER TABLE`` их не
    #: создаст у существующей базы, а ``CREATE TABLE IF NOT EXISTS`` — тем более.
    MIGRATIONS = (
        ("formulas", "latex_geom", "TEXT"),
    )

    def _migrate(self) -> None:
        for table, column, sql_type in self.MIGRATIONS:
            existing = {
                row["name"]
                for row in self.conn.execute(f"PRAGMA table_info({table})")
            }
            if column not in existing:
                self.conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"
                )

    # -------------------------------------------------------------- documents

    def upsert_document(self, **fields: Any) -> int:
        """Создаёт или обновляет документ по ``code``; возвращает doc_id."""
        code = fields["code"]
        row = self.conn.execute("SELECT doc_id FROM documents WHERE code = ?", (code,)).fetchone()
        if row is None:
            columns = ", ".join(fields)
            placeholders = ", ".join("?" for _ in fields)
            cursor = self.conn.execute(
                f"INSERT INTO documents ({columns}) VALUES ({placeholders})",
                tuple(fields.values()),
            )
            self.conn.commit()
            return int(cursor.lastrowid)
        updates = ", ".join(f"{key} = ?" for key in fields if key != "code")
        values = [value for key, value in fields.items() if key != "code"]
        if updates:
            self.conn.execute(
                f"UPDATE documents SET {updates} WHERE code = ?",
                (*values, code),
            )
            self.conn.commit()
        return int(row["doc_id"])

    def get_document(self, code: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM documents WHERE code = ?", (code,)).fetchone()

    def list_documents(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM documents ORDER BY code"))

    # ----------------------------------------------------------------- chunks

    def replace_chunks(self, doc_id: int, doc_code: str, rows: Sequence[dict[str, Any]]) -> int:
        """Полная замена чанков документа (идемпотентная переиндексация)."""
        conn = self.conn
        old = [row["chunk_id"] for row in conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,))]
        if old:
            conn.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", [(chunk_id,) for chunk_id in old])
            conn.executemany("DELETE FROM chunk_clauses WHERE chunk_id = ?", [(chunk_id,) for chunk_id in old])
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        for row in rows:
            cursor = conn.execute(
                """
                INSERT INTO chunks (doc_id, ord, kind, clause_id, section, section_title,
                                    heading, page_start, page_end, text, embed_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id, row["ord"], row["kind"], row["clause_id"], row["section"],
                    row["section_title"], row["heading"], row["page_start"], row["page_end"],
                    row["text"], row.get("embed_text"),
                ),
            )
            chunk_id = int(cursor.lastrowid)
            clause_ids = _unique(row.get("clause_ids") or ([row["clause_id"]] if row["clause_id"] else []))
            conn.execute(
                "INSERT INTO chunks_fts (text, clause_id, section_title, chunk_id, doc_code) VALUES (?, ?, ?, ?, ?)",
                (row["text"], " ".join(clause_ids), row["section_title"] or "", chunk_id, doc_code),
            )
            conn.executemany(
                "INSERT OR REPLACE INTO chunk_clauses (chunk_id, clause_id) VALUES (?, ?)",
                [(chunk_id, clause_id) for clause_id in clause_ids],
            )
        conn.commit()
        return len(rows)

    def chunks_missing_embedding(self, doc_id: int | None = None, limit: int | None = None) -> list[sqlite3.Row]:
        sql = f"""
            SELECT c.*, d.code AS doc_code, d.title AS doc_title,
                   {CLAUSE_IDS_EXPR} AS clause_ids
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.embedded_at IS NULL
        """
        params: list[Any] = []
        if doc_id is not None:
            sql += " AND c.doc_id = ?"
            params.append(doc_id)
        sql += " ORDER BY c.doc_id, c.ord"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return list(self.conn.execute(sql, params))

    def set_embedded(self, chunk_id: int, embed_text: str, point_id: str) -> None:
        self.conn.execute(
            "UPDATE chunks SET embed_text = ?, embedded_at = ?, point_id = ? WHERE chunk_id = ?",
            (embed_text, datetime.now().isoformat(timespec="seconds"), point_id, chunk_id),
        )

    def commit(self) -> None:
        self.conn.commit()

    # ---------------------------------------------------------------- figures
    def upsert_figure(self, **fields: Any) -> int:
        """Ставит картинку в таблицу (путь уникален в документе); описание не трогает."""
        conn = self.conn
        row = conn.execute(
            "SELECT figure_id FROM figures WHERE doc_id = ? AND path = ?",
            (fields["doc_id"], fields["path"]),
        ).fetchone()
        if row is None:
            columns = ", ".join(fields)
            placeholders = ", ".join("?" for _ in fields)
            cursor = conn.execute(
                f"INSERT INTO figures ({columns}) VALUES ({placeholders})",
                tuple(fields.values()),
            )
            conn.commit()
            return int(cursor.lastrowid)
        figure_id = int(row["figure_id"])
        conn.execute(
            "UPDATE figures SET page = ?, kind = ?, xref = ?, caption = ?, bbox = ? "
            "WHERE figure_id = ?",
            (
                fields.get("page"), fields.get("kind"), fields.get("xref"),
                fields.get("caption"), fields.get("bbox"), figure_id,
            ),
        )
        conn.commit()
        return figure_id

    def list_figures(
        self, doc_code: str | None = None, *, pending: bool | None = None
    ) -> list[sqlite3.Row]:
        sql = """SELECT f.*, d.code AS doc_code, d.title AS doc_title
                   FROM figures f JOIN documents d ON d.doc_id = f.doc_id
                  WHERE 1 = 1"""
        params: list[Any] = []
        if doc_code:
            sql += " AND d.code = ?"
            params.append(doc_code)
        if pending is True:
            sql += " AND f.description IS NULL"
        elif pending is False:
            sql += " AND f.description IS NOT NULL"
        sql += " ORDER BY d.code, f.page, f.figure_id"
        return list(self.conn.execute(sql, params))

    def get_figure(self, figure_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT f.*, d.code AS doc_code FROM figures f"
            " JOIN documents d ON d.doc_id = f.doc_id WHERE f.figure_id = ?",
            (figure_id,),
        ).fetchone()

    def set_figure_description(
        self, figure_id: int, text: str, by: str = "agent"
    ) -> tuple[sqlite3.Row, int | None]:
        """Записывает описание и вшивает его в чанк-носитель меткой fig:N."""
        self.conn.execute(
            "UPDATE figures SET description = ?, described_by = ?, described_at = ? "
            "WHERE figure_id = ?",
            (
                str(text).strip(), by,
                datetime.now().isoformat(timespec="seconds"), figure_id,
            ),
        )
        row = self.get_figure(figure_id)
        if row is None:
            self.conn.commit()
            raise ValueError(f"картинка id={figure_id} не найдена")
        chunk_id = self.apply_figure_note(row)
        self.conn.commit()
        return row, chunk_id

    def apply_figure_note(self, figure: sqlite3.Row) -> int | None:
        """Переписывает метку описания в тексте чанка (идемпотентно)."""
        figure_id = int(figure["figure_id"])
        description = figure["description"]
        if not description:
            return None
        chunk = self._figure_chunk(figure)
        if chunk is None:
            return None
        caption = (figure["caption"] or "").strip()
        caption = caption or f"Изображение (стр. {figure['page']})"
        note = (
            f"<!-- fig:{figure_id} -->\n"
            f"**{caption}** (стр. {figure['page']}): {description}\n"
            f"<!-- /fig:{figure_id} -->"
        )
        return self._apply_note(
            chunk, note, lambda text: _strip_figure_note(text, figure_id)
        )

    def _apply_note(self, chunk: sqlite3.Row, note: str, strip) -> int:
        """Вшивает готовую метку в чанк: текст, embed_text, FTS, устаревший вектор.

        Общий путь для описаний картинок и формул: ``embedded_at = NULL``
        означает «вектор чанка устарел» — доэмбеджит ``norms.py reembed``.
        """
        text = strip(chunk["text"] or "")
        base_embed = chunk["embed_text"] or chunk["text"] or ""
        embed_text = strip(base_embed)
        new_text = f"{text}\n\n{note}".strip()
        new_embed = f"{embed_text}\n\n{note}".strip()
        return self._write_chunk_text(chunk, new_text, new_embed)

    def _write_chunk_text(self, chunk: sqlite3.Row, new_text: str, new_embed: str) -> int:
        """Переписывает текст/embed чанка и его FTS-строку, помечая вектор устаревшим."""
        self.conn.execute(
            "UPDATE chunks SET text = ?, embed_text = ?, embedded_at = NULL "
            "WHERE chunk_id = ?",
            (new_text, new_embed, chunk["chunk_id"]),
        )
        # FTS хранит своё копие текста — меняем его тем же способом
        self.conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk["chunk_id"],))
        clause_ids = " ".join(
            r["clause_id"]
            for r in self.conn.execute(
                "SELECT clause_id FROM chunk_clauses WHERE chunk_id = ? ORDER BY rowid",
                (chunk["chunk_id"],),
            )
        )
        doc_code = self.conn.execute(
            "SELECT code FROM documents WHERE doc_id = ?", (chunk["doc_id"],)
        ).fetchone()[0]
        self.conn.execute(
            "INSERT INTO chunks_fts (text, clause_id, section_title, chunk_id, doc_code) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                new_text, clause_ids, chunk["section_title"] or "",
                chunk["chunk_id"], doc_code,
            ),
        )
        return int(chunk["chunk_id"])

    def reapply_figure_notes(self, doc_id: int) -> int:
        """Заново вшивает описания в чанки (replace_chunks их вырезал)."""
        rows = list(
            self.conn.execute(
                "SELECT * FROM figures WHERE doc_id = ? AND description IS NOT NULL",
                (doc_id,),
            )
        )
        for figure in rows:
            self.apply_figure_note(figure)
        return len(rows)

    def _figure_chunk(self, figure: sqlite3.Row) -> sqlite3.Row | None:
        """Чанк-носитель картинки: сначала по подписи, иначе по странице."""
        caption = (figure["caption"] or "")[:12]
        if caption:
            row = self.conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? AND instr(text, ?) > 0 "
                "ORDER BY LENGTH(text) LIMIT 1",
                (figure["doc_id"], caption),
            ).fetchone()
            if row is not None:
                return row
        page = figure["page"]
        if page:
            row = self.conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? AND page_start IS NOT NULL "
                "AND page_start <= ? AND page_end >= ? "
                "ORDER BY LENGTH(text) LIMIT 1",
                (figure["doc_id"], page, page),
            ).fetchone()
            if row is not None:
                return row
        return None

    # ---------------------------------------------------------------- tables
    def replace_tables(self, doc_id: int, rows: Sequence[dict[str, Any]]) -> int:
        """Полная замена структурированных таблиц документа (идемпотентно).

        Чанк-носитель ищется по характерной ячейке (``match_text``): самая
        короткая строка чанка, содержащая подстроку, — так таблица цепляется
        за свой же чанк ``kind=table``, а не за соседний.
        """
        from .tables import parse_number

        conn = self.conn
        conn.execute("DELETE FROM tables WHERE doc_id = ?", (doc_id,))
        for row in rows:
            chunk_id = self._chunk_by_text(doc_id, row.get("match_text") or "")
            header = list(row.get("header") or [])
            data_rows = [list(item) for item in row.get("rows") or []]
            col_count = max([len(header)] + [len(item) for item in data_rows], default=0)
            cursor = conn.execute(
                "INSERT INTO tables (doc_id, chunk_id, ord, caption, page_start,"
                " page_end, header, row_count, col_count)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc_id, chunk_id, row["ord"], row.get("caption") or "",
                    row.get("page_start"), row.get("page_end"),
                    json.dumps(header, ensure_ascii=False),
                    len(data_rows), col_count,
                ),
            )
            table_id = int(cursor.lastrowid)
            cells = [(0, col, value) for col, value in enumerate(header)]
            for row_idx, data_row in enumerate(data_rows, start=1):
                cells.extend((row_idx, col, value) for col, value in enumerate(data_row))
            conn.executemany(
                "INSERT INTO table_cells (table_id, row_idx, col_idx, raw, num)"
                " VALUES (?, ?, ?, ?, ?)",
                [
                    (table_id, row_idx, col, str(value), parse_number(value))
                    for row_idx, col, value in cells
                ],
            )
        conn.commit()
        return len(rows)

    def _chunk_by_text(self, doc_id: int, needle: str) -> int | None:
        """Чанк-носитель по подстроке; None — подстрока короткая или не найдена."""
        needle = needle.strip()
        if len(needle) < 4:
            return None
        row = self.conn.execute(
            "SELECT chunk_id FROM chunks WHERE doc_id = ? AND instr(text, ?) > 0"
            " ORDER BY LENGTH(text) LIMIT 1",
            (doc_id, needle),
        ).fetchone()
        return int(row["chunk_id"]) if row else None

    def list_tables(self, doc_code: str | None = None) -> list[sqlite3.Row]:
        sql = """SELECT t.*, d.code AS doc_code
                   FROM tables t JOIN documents d ON d.doc_id = t.doc_id
                  WHERE 1 = 1"""
        params: list[Any] = []
        if doc_code:
            sql += " AND d.code = ?"
            params.append(doc_code)
        sql += " ORDER BY d.code, t.ord"
        return list(self.conn.execute(sql, params))

    def get_table(self, table_id: int) -> dict[str, Any] | None:
        """Шапка + ячейки строк данных: raw (источник правды) и num (число)."""
        head = self.conn.execute(
            "SELECT t.*, d.code AS doc_code FROM tables t"
            " JOIN documents d ON d.doc_id = t.doc_id WHERE t.table_id = ?",
            (table_id,),
        ).fetchone()
        if head is None:
            return None
        grid: dict[int, dict[int, dict[str, Any]]] = {}
        for cell in self.conn.execute(
            "SELECT row_idx, col_idx, raw, num FROM table_cells"
            " WHERE table_id = ? ORDER BY row_idx, col_idx",
            (table_id,),
        ):
            grid.setdefault(int(cell["row_idx"]), {})[int(cell["col_idx"])] = {
                "raw": cell["raw"],
                "num": cell["num"],
            }
        data_rows = [
            [
                grid.get(row_idx, {}).get(col, {"raw": "", "num": None})
                for col in range(int(head["col_count"]))
            ]
            for row_idx in range(1, int(head["row_count"]) + 1)
        ]
        return {
            "table_id": int(head["table_id"]),
            "doc_code": head["doc_code"],
            "caption": head["caption"] or "",
            "page_start": head["page_start"],
            "page_end": head["page_end"],
            "chunk_id": head["chunk_id"],
            "header": json.loads(head["header"] or "[]"),
            "row_count": head["row_count"],
            "col_count": head["col_count"],
            "rows": data_rows,
        }

    # -------------------------------------------------------------- formulas
    def upsert_formula(self, **fields: Any) -> int:
        """Ставит формулу в таблицу (путь уникален в документе); описание не трогает."""
        conn = self.conn
        row = conn.execute(
            "SELECT formula_id FROM formulas WHERE doc_id = ? AND path = ?",
            (fields["doc_id"], fields["path"]),
        ).fetchone()
        if row is None:
            columns = ", ".join(fields)
            placeholders = ", ".join("?" for _ in fields)
            cursor = conn.execute(
                f"INSERT INTO formulas ({columns}) VALUES ({placeholders})",
                tuple(fields.values()),
            )
            conn.commit()
            return int(cursor.lastrowid)
        formula_id = int(row["formula_id"])
        conn.execute(
            "UPDATE formulas SET tag = ?, page = ?, bbox = ?, anchor = ?,"
            " latex_geom = ? WHERE formula_id = ?",
            (
                fields.get("tag"), fields.get("page"), fields.get("bbox"),
                fields.get("anchor"), fields.get("latex_geom"), formula_id,
            ),
        )
        conn.commit()
        return formula_id

    def drop_missing_formulas(self, doc_id: int, keep: Sequence[str]) -> None:
        """Удаляет формулы документа, чьих файлов больше нет (PDF изменился).

        Пустой ``keep`` ничего не трогает: нулевой результат извлечения —
        не повод стирать уже записанные описания.
        """
        if not keep:
            return
        placeholders = ",".join("?" for _ in keep)
        self.conn.execute(
            f"DELETE FROM formulas WHERE doc_id = ? AND path NOT IN ({placeholders})",
            (doc_id, *keep),
        )

    def list_formulas(
        self, doc_code: str | None = None, *, pending: bool | None = None
    ) -> list[sqlite3.Row]:
        sql = """SELECT f.*, d.code AS doc_code, d.title AS doc_title
                   FROM formulas f JOIN documents d ON d.doc_id = f.doc_id
                  WHERE 1 = 1"""
        params: list[Any] = []
        if doc_code:
            sql += " AND d.code = ?"
            params.append(doc_code)
        empty = ("(COALESCE(f.description, '') = '' AND COALESCE(f.expr, '') = ''"
                 " AND COALESCE(f.latex, '') = '')")
        if pending is True:
            sql += f" AND {empty}"
        elif pending is False:
            sql += f" AND NOT {empty}"
        sql += " ORDER BY d.code, f.page, f.formula_id"
        return list(self.conn.execute(sql, params))

    def get_formula(self, formula_id: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT f.*, d.code AS doc_code FROM formulas f"
            " JOIN documents d ON d.doc_id = f.doc_id WHERE f.formula_id = ?",
            (formula_id,),
        ).fetchone()

    def set_formula_data(
        self,
        formula_id: int,
        *,
        latex: str | None = None,
        expr: str | None = None,
        params: dict[str, Any] | None = None,
        description: str | None = None,
        by: str = "agent",
    ) -> tuple[sqlite3.Row, int | None]:
        """Заполняет распознавание формулы и вшивает метку в чанк-носитель.

        Не переданные поля остаются прежними; вставка метки помечает чанк
        «без вектора» (``embedded_at = NULL``) — после описания нужен
        ``norms.py reembed``.
        """
        row = self.get_formula(formula_id)
        if row is None:
            raise ValueError(f"формула id={formula_id} не найдена")
        merged = {
            "latex": row["latex"] if latex is None else str(latex).strip(),
            "expr": row["expr"] if expr is None else str(expr).strip(),
            "params": (
                row["params"] if params is None
                else json.dumps(params, ensure_ascii=False)
            ),
            "description": (
                row["description"] if description is None
                else str(description).strip()
            ),
        }
        self.conn.execute(
            "UPDATE formulas SET latex = ?, expr = ?, params = ?, description = ?,"
            " described_by = ?, described_at = ? WHERE formula_id = ?",
            (
                merged["latex"], merged["expr"], merged["params"],
                merged["description"], by,
                datetime.now().isoformat(timespec="seconds"), formula_id,
            ),
        )
        row = self.get_formula(formula_id)
        chunk_id = self.apply_formula_note(row)
        self.conn.commit()
        return row, chunk_id

    def apply_formula_note(self, formula: sqlite3.Row) -> int | None:
        """Переписывает метку описания формулы в тексте чанка (идемпотентно)."""
        note = _formula_note(formula)
        if note is None:
            return None
        chunk = self._formula_chunk(formula)
        if chunk is None:
            return None
        return self._apply_note(
            chunk, note,
            lambda text: _strip_formula_note(text, int(formula["formula_id"])),
        )

    def reapply_formula_notes(self, doc_id: int) -> int:
        """Заново вшивает описания формул в чанки (replace_chunks их вырезал).

        Сначала все метки ``fml`` снимаются с чанков документа: чанк-носитель
        может смениться (тег «(А.2)» встречается и в ссылках на другие пункты),
        и старый носитель не должен удерживать чужое описание.
        """
        for chunk in list(
            self.conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ?"
                " AND instr(text, '<!-- fml:') > 0",
                (doc_id,),
            )
        ):
            cleaned_text = _strip_all_formula_notes(chunk["text"] or "")
            cleaned_embed = _strip_all_formula_notes(chunk["embed_text"] or chunk["text"] or "")
            if cleaned_text != (chunk["text"] or ""):
                self._write_chunk_text(chunk, cleaned_text, cleaned_embed)
        rows = list(
            self.conn.execute(
                "SELECT * FROM formulas WHERE doc_id = ?"
                " AND (description IS NOT NULL OR latex IS NOT NULL"
                "      OR expr IS NOT NULL)",
                (doc_id,),
            )
        )
        for formula in rows:
            self.apply_formula_note(formula)
        return len(rows)

    def _formula_chunk(self, formula: sqlite3.Row) -> sqlite3.Row | None:
        """Чанк-носитель формулы: тег на странице формулы > тег где угодно > страница.

        «(А.2)» встречается в ссылках по всему документу («по формуле (А.2)…») —
        брать самый короткий чанк с тегом значило бы вшивать описание в чужой
        пункт; привязка к странице формулы отсекает такие ссылки.
        """
        tag = (formula["tag"] or "").strip()
        page = formula["page"]
        if tag:
            needle = f"({tag})"
            if page:
                row = self.conn.execute(
                    "SELECT * FROM chunks WHERE doc_id = ?"
                    " AND page_start IS NOT NULL AND page_start <= ? AND page_end >= ?"
                    " AND instr(text, ?) > 0"
                    " ORDER BY LENGTH(text) LIMIT 1",
                    (formula["doc_id"], page, page, needle),
                ).fetchone()
                if row is not None:
                    return row
            row = self.conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? AND instr(text, ?) > 0"
                " ORDER BY LENGTH(text) LIMIT 1",
                (formula["doc_id"], needle),
            ).fetchone()
            if row is not None:
                return row
        if page:
            row = self.conn.execute(
                "SELECT * FROM chunks WHERE doc_id = ? AND page_start IS NOT NULL"
                " AND page_start <= ? AND page_end >= ?"
                " ORDER BY LENGTH(text) LIMIT 1",
                (formula["doc_id"], page, page),
            ).fetchone()
            if row is not None:
                return row
        return None

    # ------------------------------------------------------------------ search

    def search_fts(
        self, query: str, *, doc_code: str | None = None, limit: int = 10
    ) -> list[ChunkRow]:
        """BM25-поиск; запрос превращается в ``"tok1" OR "tok2"`` — фразы не ждём."""
        tokens = _TOKEN.findall(str(query))
        if not tokens:
            return []
        match = " OR ".join(f'"{token}"' for token in tokens)
        sql = f"""
            SELECT {SELECT_COLUMNS}, bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE chunks_fts MATCH ?
        """
        params: list[Any] = [match]
        if doc_code:
            sql += " AND d.code = ?"
            params.append(doc_code)
        sql += " ORDER BY rank LIMIT ?"
        params.append(int(limit))
        return [ChunkRow.from_row(row, with_rank=True) for row in self.conn.execute(sql, params)]

    def find_clause(
        self, doc_code: str, clause_id: str, *, limit: int = 50
    ) -> list[ChunkRow]:
        """Точная ссылка на пункт: ``('СП 60.13330.2020', '7.6.9')``.

        Ищет по ``chunk_clauses``, а не по ``chunks.clause_id``: склейка мелких
        пунктов оставляет в чанке только номер первого, поэтому пункт 7.6.9
        может лежать в чанке с primary-номером 7.6.8.

        К базовому коду подмешиваются документы-правки того же свода — коды вида
        ``'СП 60.13330.2020 (изм. 4)'``: точная ссылка обязана отдавать
        актуальную редакцию, а не только исходный текст. Правки идут после
        базового документа (основной код — префикс, сортировка по коду сохраняет
        порядок «сначала оригинал, затем изменения»).
        """
        pattern = _like_escape(doc_code) + " (%"
        sql = f"""
            SELECT {SELECT_COLUMNS}
            FROM chunk_clauses cc
            JOIN chunks c ON c.chunk_id = cc.chunk_id
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE cc.clause_id = ?
              AND (d.code = ? OR d.code LIKE ? ESCAPE '\\')
            ORDER BY d.code, c.ord LIMIT ?
        """
        return [
            ChunkRow.from_row(row)
            for row in self.conn.execute(sql, (clause_id, doc_code, pattern, limit))
        ]

    def find_section(self, doc_code: str, section: str, *, limit: int = 50) -> list[ChunkRow]:
        sql = f"""
            SELECT {SELECT_COLUMNS}
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.section = ?
              AND (d.code = ? OR d.code LIKE ? ESCAPE '\\')
            ORDER BY d.code, c.ord LIMIT ?
        """
        pattern = _like_escape(doc_code) + " (%"
        return [
            ChunkRow.from_row(row)
            for row in self.conn.execute(sql, (section, doc_code, pattern, limit))
        ]

    def get_chunks_by_ids(self, chunk_ids: Iterable[int]) -> dict[int, ChunkRow]:
        ids = list(chunk_ids)
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        sql = f"""
            SELECT {SELECT_COLUMNS}
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.chunk_id IN ({placeholders})
        """
        rows = self.conn.execute(sql, ids).fetchall()
        return {row["chunk_id"]: ChunkRow.from_row(row) for row in rows}

    # ------------------------------------------------------------------- stats

    def stats(self) -> dict[str, Any]:
        docs = self.conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunks = self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        embedded = self.conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE embedded_at IS NOT NULL"
        ).fetchone()["n"]
        chars = self.conn.execute("SELECT COALESCE(SUM(LENGTH(text)), 0) AS n FROM chunks").fetchone()["n"]
        clauses = self.conn.execute(
            "SELECT COUNT(DISTINCT clause_id) AS n FROM chunk_clauses"
        ).fetchone()["n"]
        return {
            "documents": docs,
            "chunks": chunks,
            "clauses": clauses,
            "embedded": embedded,
            "pending_embedding": chunks - embedded,
            "figures": self.conn.execute(
                "SELECT COUNT(*) AS n FROM figures"
            ).fetchone()["n"],
            "tables": self.conn.execute(
                "SELECT COUNT(*) AS n FROM tables"
            ).fetchone()["n"],
            "formulas": self.conn.execute(
                "SELECT COUNT(*) AS n FROM formulas"
            ).fetchone()["n"],
            "pending_formulas": self.conn.execute(
                "SELECT COUNT(*) AS n FROM formulas"
                " WHERE COALESCE(description, '') = ''"
                " AND COALESCE(expr, '') = '' AND COALESCE(latex, '') = ''"
            ).fetchone()["n"],
            "chars": chars,
            "db_path": str(self.db_path),
        }

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None
