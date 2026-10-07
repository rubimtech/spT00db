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
"""

_TOKEN = re.compile(r"\w+", re.UNICODE)


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
        self.conn.commit()

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
                                    heading, page_start, page_end, text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id, row["ord"], row["kind"], row["clause_id"], row["section"],
                    row["section_title"], row["heading"], row["page_start"], row["page_end"],
                    row["text"],
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
        """
        sql = f"""
            SELECT {SELECT_COLUMNS}
            FROM chunk_clauses cc
            JOIN chunks c ON c.chunk_id = cc.chunk_id
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE d.code = ? AND cc.clause_id = ?
            ORDER BY c.ord LIMIT ?
        """
        return [ChunkRow.from_row(row) for row in self.conn.execute(sql, (doc_code, clause_id, limit))]

    def find_section(self, doc_code: str, section: str, *, limit: int = 50) -> list[ChunkRow]:
        sql = f"""
            SELECT {SELECT_COLUMNS}
            FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
            WHERE d.code = ? AND c.section = ?
            ORDER BY c.ord LIMIT ?
        """
        return [ChunkRow.from_row(row) for row in self.conn.execute(sql, (doc_code, section, limit))]

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
            "chars": chars,
            "db_path": str(self.db_path),
        }

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None
