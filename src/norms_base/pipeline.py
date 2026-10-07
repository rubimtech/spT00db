"""Конвейер инжеста одного документа: PDF -> Markdown -> SQLite -> Qdrant.

Шаги и что на каждом теряется бы молча:

1. скачивание — антибот-заглушка принимается за PDF только если нет магии ``%PDF-``;
2. конверсия — колонтитулы/номера страниц чистятся построчно, до склейки;
3. чанкинг — границы по пунктам, страницы берутся из карты смещений;
4. запись в SQLite — замена чанков идемпотентна, статус ``embedded_at`` даёт
   возобновление после обрыва;
5. эмбеддинг + upsert в Qdrant — старые точки документа удаляются, чтобы
   после правок не оставалось отвечающих на запросы чанков-призраков.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .chunk import build_embed_text, chunk_markdown
from .config import MD_DIR, PDF_DIR, ensure_dirs
from .convert import pdf_to_markdown
from .download import fetch_pdf
from .embed import Embedder
from .storage import Store, parse_clause_ids
from .vectorstore import VectorStore, point_id_for


@dataclass(slots=True)
class AddResult:
    doc_code: str
    pages: int
    chars: int
    chunks: int
    embedded: int
    vectors: int
    seconds: float
    md_path: Path
    pdf_path: Path


def add_document(
    code: str,
    title: str,
    *,
    url: str | None = None,
    pdf_path: str | Path | None = None,
    store: Store | None = None,
    vector_store: VectorStore | None = None,
    embedder: Embedder | None = None,
    batch_size: int = 64,
    verbose: bool = True,
) -> AddResult:
    """Индексирует один документ от PDF до векторов. Идемпотентно по ``code``."""
    if not url and not pdf_path:
        raise ValueError("нужен --url или --pdf")
    ensure_dirs()
    started = time.perf_counter()

    own_store = store is None
    store = store or Store(_default_db())
    store.init_schema()
    vector_store = vector_store or VectorStore()

    # 1. PDF
    if pdf_path is not None:
        source = Path(pdf_path)
        if not source.exists():
            raise FileNotFoundError(f"PDF не найден: {source}")
        import hashlib

        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        fetched_path, source_note = source, str(source)
    else:
        fetched = fetch_pdf(str(url), PDF_DIR)
        fetched_path, sha, source_note = fetched.path, fetched.sha256, str(url)
        if verbose:
            print(f"[pdf] {fetched_path.name} | {fetched.size} байт | sha256 {sha[:16]}", flush=True)

    # 2. Markdown
    doc = pdf_to_markdown(fetched_path, code)
    md_path = MD_DIR / f"{_slug(code)}.md"
    md_path.write_text(doc.text, encoding="utf-8")
    if verbose:
        print(f"[md] {len(doc.text)} символов, {doc.page_count} страниц -> {md_path.name}", flush=True)

    # 3. Чанки (+ номера страниц из карты смещений)
    chunks = chunk_markdown(doc.text)
    rows = [
        {
            "ord": chunk.ord,
            "kind": chunk.kind,
            "clause_id": chunk.clause_id,
            "clause_ids": list(chunk.clauses),
            "section": chunk.section,
            "section_title": chunk.section_title,
            "heading": chunk.heading,
            "page_start": doc.page_for(chunk.start),
            "page_end": doc.page_for(max(chunk.end - 1, chunk.start)),
            "text": chunk.text,
        }
        for chunk in chunks
    ]
    if verbose:
        clauses = {cid for row in rows for cid in row["clause_ids"]}
        shared = sum(1 for row in rows if len(row["clause_ids"]) > 1)
        print(
            f"[chunk] {len(rows)} чанков, {len(clauses)} уникальных пунктов "
            f"({shared} чанков несут больше одного пункта)",
            flush=True,
        )

    # 4. SQLite
    existed = store.get_document(code) is not None
    doc_id = store.upsert_document(
        code=code,
        title=title,
        url=url,
        sha256=sha,
        source=source_note,
        pdf_path=str(fetched_path),
        md_path=str(md_path),
        pages=doc.page_count,
        chars=len(doc.text),
        chunk_count=len(rows),
        downloaded_at=datetime.now().isoformat(timespec="seconds"),
        converted_at=datetime.now().isoformat(timespec="seconds"),
    )
    store.replace_chunks(doc_id, code, rows)

    # 5. Векторы: старые точки документа убираем, новые эмбедим и грузим
    if not vector_store.available():
        vector_store.ensure_collection()
    if existed:
        vector_store.delete_document(code)

    engine = embedder or Embedder()
    pending = store.chunks_missing_embedding(doc_id)
    embedded = 0
    vectors_uploaded = 0
    buffer: list[dict[str, Any]] = []
    batch_started = time.perf_counter()
    for index, row in enumerate(pending, start=1):
        chunk = chunks[row["ord"]]
        embed_text = build_embed_text(chunk, code, title)
        vector = engine.embed_one(embed_text)
        point_id = point_id_for(code, row["ord"])
        store.set_embedded(row["chunk_id"], embed_text, point_id)
        buffer.append(
            {
                "id": point_id,
                "vector": vector,
                "payload": {
                    "chunk_id": row["chunk_id"],
                    "doc_code": code,
                    "doc_title": title,
                    "clause_id": row["clause_id"],
                    "clause_ids": parse_clause_ids(row["clause_ids"]),
                    "section": row["section"],
                    "section_title": row["section_title"],
                    "heading": row["heading"],
                    "kind": row["kind"],
                    "page_start": row["page_start"],
                    "page_end": row["page_end"],
                    "ord": row["ord"],
                    "text": row["text"],
                },
            }
        )
        embedded += 1
        if len(buffer) >= max(batch_size, 1):
            vectors_uploaded += vector_store.upsert(buffer)
            store.commit()
            buffer = []
            if verbose and index % 128 == 0:
                rate = index / max(time.perf_counter() - batch_started, 1e-9)
                print(f"[embed] {index}/{len(pending)} | {rate:.1f} текстов/сек", flush=True)
    if buffer:
        vectors_uploaded += vector_store.upsert(buffer)
    store.commit()

    if own_store:
        store.close()

    result = AddResult(
        doc_code=code,
        pages=doc.page_count,
        chars=len(doc.text),
        chunks=len(rows),
        embedded=embedded,
        vectors=vectors_uploaded,
        seconds=time.perf_counter() - started,
        md_path=md_path,
        pdf_path=fetched_path,
    )
    if verbose:
        print(
            f"[done] {code}: {result.chunks} чанков, {result.embedded} векторов, "
            f"{result.vectors} точек Qdrant, {result.seconds:.1f} с",
            flush=True,
        )
    return result


def reembed_pending(
    store: Store,
    vector_store: VectorStore | None = None,
    embedder: Embedder | None = None,
    *,
    batch_size: int = 64,
    limit: int | None = None,
    verbose: bool = True,
) -> int:
    """Доэмбедживает чанки с ``embedded_at IS NULL`` (возобновление после обрыва)."""
    vector_store = vector_store or VectorStore()
    if not vector_store.available():
        vector_store.ensure_collection()
    engine = embedder or Embedder()
    pending = store.chunks_missing_embedding(limit=limit)
    if not pending:
        if verbose:
            print("[embed] нечего эмбедить", flush=True)
        return 0

    documents = {row["doc_id"]: (row["doc_code"], row["doc_title"]) for row in pending}
    buffer: list[dict[str, Any]] = []
    done = 0
    for row in pending:
        doc_code, doc_title = documents[row["doc_id"]]
        embed_text = row["embed_text"] or _fallback_embed_text(row, doc_code, doc_title)
        vector = engine.embed_one(embed_text)
        point_id = row["point_id"] or point_id_for(doc_code, row["ord"])
        store.set_embedded(row["chunk_id"], embed_text, point_id)
        buffer.append(
            {
                "id": point_id,
                "vector": vector,
                "payload": {
                    "chunk_id": row["chunk_id"],
                    "doc_code": doc_code,
                    "doc_title": doc_title,
                    "clause_id": row["clause_id"],
                    "clause_ids": parse_clause_ids(row["clause_ids"]),
                    "section": row["section"],
                    "section_title": row["section_title"],
                    "heading": row["heading"],
                    "kind": row["kind"],
                    "page_start": row["page_start"],
                    "page_end": row["page_end"],
                    "ord": row["ord"],
                    "text": row["text"],
                },
            }
        )
        done += 1
        if len(buffer) >= max(batch_size, 1):
            vector_store.upsert(buffer)
            store.commit()
            buffer = []
    if buffer:
        vector_store.upsert(buffer)
    store.commit()
    if verbose:
        print(f"[embed] доэмбеджено {done} чанков", flush=True)
    return done


def status(store: Store, vector_store: VectorStore | None = None) -> dict[str, Any]:
    """Сверка SQLite с Qdrant: расхождение означает, что точки устарели."""
    vector_store = vector_store or VectorStore()
    info = store.stats()
    try:
        info["qdrant_points"] = vector_store.count() if vector_store.available() else None
    except Exception as exc:  # noqa: BLE001
        info["qdrant_points"] = f"недоступно: {type(exc).__name__}"
    info["collection"] = vector_store.collection
    info["qdrant_url"] = vector_store.url
    if isinstance(info.get("qdrant_points"), int):
        info["sync_delta"] = info["qdrant_points"] - info["embedded"]
    return info


def _fallback_embed_text(row: Any, doc_code: str, doc_title: str) -> str:
    head = f"{doc_code}. {doc_title}."
    section = row["section"] or ""
    if section:
        head += f" Раздел {section}."
    return f"{head} {row['text']}"


def _slug(code: str) -> str:
    return re.sub(r"[^\w.]+", "_", code, flags=re.UNICODE).strip("_") or "document"


def _default_db() -> Path:
    from .config import DB_PATH

    return DB_PATH
