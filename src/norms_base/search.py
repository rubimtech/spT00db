"""API поиска по базе нормативов для агента-проверяющего.

Три способа найти пункт, и у каждого своя задача:

* :func:`clause` — точная ссылка («п. 7.6.9 СП 60.13330.2020»): SQLite,
  никакой семантики, порядок как в документе;
* :func:`keyword` — лексический BM25 по FTS5: ловит точные формулировки
  («предел огнестойкости транзитных воздуховодов»);
* :func:`semantic` — вектор bge-m3: ловит синонимы и перефразировки
  («куда можно выбрасывать воздух из вентиляции»);
* :func:`hybrid` — RRF-слияние первых двух: рабочий режим по умолчанию,
  потому что проверяющий агент обычно не знает, совпали ли слова нормы
  с тем, что написано в проекте.

Все ответы несут готовую цитату: пункт, раздел, страницы и текст — чтобы
вывод о несоответствии опирался на ссылку, а не на пересказ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import DB_PATH
from .embed import Embedder
from .storage import ChunkRow, Store
from .vectorstore import VectorStore

RRF_K = 60


@dataclass(slots=True)
class Hit:
    chunk_id: int
    doc_code: str
    doc_title: str
    clause_id: str | None
    section: str | None
    section_title: str | None
    heading: str | None
    page_start: int | None
    page_end: int | None
    text: str
    score: float
    #: все пункты чанка: склейка мелких пунктов кладёт 7.6.8 и 7.6.9 в один чанк
    clause_ids: tuple[str, ...] = ()
    sources: list[str] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: ChunkRow, score: float, sources: list[str]) -> "Hit":
        return cls(
            chunk_id=row.chunk_id,
            doc_code=row.doc_code,
            doc_title=row.doc_title,
            clause_id=row.clause_id,
            section=row.section,
            section_title=row.section_title,
            heading=row.heading,
            page_start=row.page_start,
            page_end=row.page_end,
            text=row.text,
            score=score,
            clause_ids=row.clause_ids,
            sources=sources,
        )

    def as_dict(self, text_limit: int | None = None) -> dict[str, Any]:
        text = self.text if text_limit is None else self.text[:text_limit]
        return {
            "chunk_id": self.chunk_id,
            "doc_code": self.doc_code,
            "doc_title": self.doc_title,
            "clause_id": self.clause_id,
            "clause_ids": list(self.clause_ids),
            "section": self.section,
            "section_title": self.section_title,
            "heading": self.heading,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "score": round(self.score, 4),
            "sources": self.sources,
            "text": text,
        }

    @property
    def cite(self) -> str:
        where = f"{self.doc_code}"
        if self.clause_id:
            where += f", п. {self.clause_id}"
        if self.page_end and self.page_start and self.page_end != self.page_start:
            where += f", с. {self.page_start}-{self.page_end}"
        elif self.page_start:
            where += f", с. {self.page_start}"
        return where


def _open(store: Store | None) -> tuple[Store, bool]:
    if store is not None:
        store.init_schema()
        return store, False
    opened = Store(DB_PATH)
    opened.init_schema()
    return opened, True


def clause(doc_code: str, clause_id: str, *, store: Store | None = None) -> list[Hit]:
    """Точный пункт: ``clause('СП 60.13330.2020', '7.6.9')``.

    Чанк может нести несколько пунктов (склейка мелких), поэтому в выдаче
    ``clause_id`` — запрошенный номер (корректная ссылка в цитате), а соседние
    пункты того же чанка перечислены в ``clause_ids``.
    """
    opened, own = _open(store)
    try:
        rows = opened.find_clause(doc_code, clause_id)
        hits = [Hit.from_row(row, score=1.0, sources=["exact"]) for row in rows]
        for hit in hits:
            if clause_id in hit.clause_ids:
                hit.clause_id = clause_id
        return hits
    finally:
        if own:
            opened.close()


def section(doc_code: str, section: str, *, store: Store | None = None) -> list[Hit]:
    """Все чанки раздела (в порядке документа) — для обзора «что требует раздел 7»."""
    opened, own = _open(store)
    try:
        rows = opened.find_section(doc_code, section)
        return [Hit.from_row(row, score=1.0, sources=["exact"]) for row in rows]
    finally:
        if own:
            opened.close()


def keyword(query: str, *, k: int = 10, doc_code: str | None = None, store: Store | None = None) -> list[Hit]:
    """Лексический BM25 по FTS5."""
    opened, own = _open(store)
    try:
        rows = opened.search_fts(query, doc_code=doc_code, limit=k)
        hits = []
        for rank, row in enumerate(rows, start=1):
            # bm25 в SQLite: меньше лучше; в RRF-оценку превращает rank-порядок
            hits.append(Hit.from_row(row, score=1.0 / (RRF_K + rank), sources=["fts"]))
        return hits
    finally:
        if own:
            opened.close()


def semantic(
    query: str,
    *,
    k: int = 10,
    doc_code: str | None = None,
    store: Store | None = None,
    vector_store: VectorStore | None = None,
    embedder: Embedder | None = None,
    score_threshold: float | None = None,
) -> list[Hit]:
    """Векторный поиск bge-m3 по Qdrant."""
    opened, own = _open(store)
    vs = vector_store or VectorStore()
    try:
        engine = embedder or Embedder()
        vector = engine.embed_one(query)
        hits_q = vs.search(vector, limit=k, doc_code=doc_code, score_threshold=score_threshold)
        if not hits_q:
            return []
        rows = opened.get_chunks_by_ids([int(hit.payload["chunk_id"]) for hit in hits_q])
        result: list[Hit] = []
        for rank, hit in enumerate(hits_q, start=1):
            row = rows.get(int(hit.payload.get("chunk_id", -1)))
            if row is None:
                # точка есть, чанка нет: отдаём хотя бы payload, но помечаем источник
                payload = hit.payload
                result.append(
                    Hit(
                        chunk_id=int(payload.get("chunk_id", -1)),
                        doc_code=str(payload.get("doc_code", "")),
                        doc_title=str(payload.get("doc_title", "")),
                        clause_id=payload.get("clause_id"),
                        clause_ids=tuple(str(item) for item in (payload.get("clause_ids") or ())),
                        section=payload.get("section"),
                        section_title=payload.get("section_title"),
                        heading=payload.get("heading"),
                        page_start=payload.get("page_start"),
                        page_end=payload.get("page_end"),
                        text=str(payload.get("text", "")),
                        score=hit.score,
                        sources=["vector", "payload-only"],
                    )
                )
                continue
            result.append(
                Hit.from_row(
                    row,
                    score=hit.score,
                    sources=[f"vector:{rank}"],
                )
            )
        return result
    finally:
        if own:
            opened.close()


def hybrid(
    query: str,
    *,
    k: int = 8,
    doc_code: str | None = None,
    store: Store | None = None,
    vector_store: VectorStore | None = None,
    embedder: Embedder | None = None,
    candidates: int | None = None,
) -> list[Hit]:
    """RRF-слияние векторной и лексической выдач (рабочий режим)."""
    opened, own = _open(store)
    try:
        pool = int(candidates or max(k * 3, 15))
        vs = vector_store or VectorStore()
        engine = embedder or Embedder()
        vector = engine.embed_one(query)
        vec_hits = vs.search(vector, limit=pool, doc_code=doc_code)
        fts_rows = opened.search_fts(query, doc_code=doc_code, limit=pool)

        scores: dict[int, float] = {}
        meta: dict[int, tuple[ChunkRow | None, dict[str, Any] | None]] = {}
        for rank, hit in enumerate(vec_hits, start=1):
            chunk_id = int(hit.payload.get("chunk_id", -1))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            meta.setdefault(chunk_id, (None, hit.payload))
        for rank, row in enumerate(fts_rows, start=1):
            scores[row.chunk_id] = scores.get(row.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            meta.setdefault(row.chunk_id, (row, None))

        rows = opened.get_chunks_by_ids([cid for cid in scores if meta.get(cid, (None, None))[0] is None])
        hits: list[Hit] = []
        for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]:
            row, payload = meta[chunk_id]
            if row is None:
                row = rows.get(chunk_id)
            if row is None:
                continue
            sources: list[str] = []
            if any(cid == chunk_id for cid, _ in ((int(h.payload.get("chunk_id", -1)), 0) for h in vec_hits)):
                sources.append("vector")
            if any(r.chunk_id == chunk_id for r in fts_rows):
                sources.append("fts")
            hits.append(Hit.from_row(row, score=score, sources=sources or ["hybrid"]))
        return hits
    finally:
        if own:
            opened.close()


def docs(*, store: Store | None = None) -> list[dict[str, Any]]:
    """Реестр проиндексированных документов (для агента: что вообще есть в базе)."""
    opened, own = _open(store)
    try:
        return [
            {
                "code": row["code"],
                "title": row["title"],
                "pages": row["pages"],
                "chunks": row["chunk_count"],
                "url": row["url"],
                "indexed_at": row["indexed_at"],
            }
            for row in opened.list_documents()
        ]
    finally:
        if own:
            opened.close()


def get_text(chunk_id: int, *, store: Store | None = None) -> str | None:
    """Полный текст чанка по id (выдача агента обрезается, цитата — нет)."""
    opened, own = _open(store)
    try:
        rows = opened.get_chunks_by_ids([chunk_id])
        return rows[chunk_id].text if chunk_id in rows else None
    finally:
        if own:
            opened.close()
