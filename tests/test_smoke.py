"""Smoke-тесты конвейера: конверсия фикстуры -> чанки -> FTS -> вектор -> hybrid -> статус.

Всё считается на тестовой базе (tmp) и тестовой коллекции Qdrant ``norm_docs_test``,
рабочая ``data/normbase.db`` и коллекция ``norm_docs`` не трогаются.
"""

from __future__ import annotations

import re

import pytest

from conftest import CODE, require_model
from norms_base import pipeline, search
from norms_base.convert import pdf_to_markdown

# ---------------------------------------------------------------- конверсия


def test_convert_keeps_text_but_drops_headers(fixture_pdf):
    doc = pdf_to_markdown(fixture_pdf, CODE)
    lines = [line.strip() for line in doc.text.splitlines()]
    assert doc.page_count == 4
    assert doc.text, "конверсия вернула пустой текст"
    # колонтитул (оба вида: с отметкой редакции и без) и номера страниц вычищены
    assert CODE not in lines, "колонтитул остался в тексте"
    assert not any("проект, вторая редакция" in line for line in lines)
    assert not any(re.fullmatch(r"\d{1,2}", line) for line in lines), "номер страницы остался"
    # мусора Firecrawl-класса быть не должно: слова не порваны посередине
    assert "отоления" not in doc.text and "ventilляции" not in doc.text


def test_convert_joins_torn_paragraph(fixture_pdf):
    """Абзац, разорванный границей страниц 2-3, должен склеиться в одно предложение."""
    doc = pdf_to_markdown(fixture_pdf, CODE)
    assert "устанавливается по назначению помещения" in doc.text.replace("\n", " ")


def test_convert_page_map(fixture_pdf):
    doc = pdf_to_markdown(fixture_pdf, CODE)
    assert doc.page_for(doc.text.index("7.6.9")) == 2
    assert doc.page_for(doc.text.index("Приложение П")) == 3


def test_convert_keeps_table_as_markdown(fixture_pdf):
    doc = pdf_to_markdown(fixture_pdf, CODE)
    rows = [line for line in doc.text.splitlines() if line.strip().startswith("|")]
    assert len(rows) >= 8, f"таблица не распознана, строк с '|': {len(rows)}"
    assert any("Кухня с газовой плитой" in row for row in rows)


# ------------------------------------------------------------------ чанки


def test_chunks_bound_to_clauses_and_tables(fixture_pdf):
    from norms_base.chunk import chunk_markdown

    doc = pdf_to_markdown(fixture_pdf, CODE)
    chunks = chunk_markdown(doc.text)
    assert chunks, "чанков нет"
    by_clause = {cid: chunk for chunk in chunks for cid in chunk.clauses}
    assert "7.6.9" in by_clause, f"пункт 7.6.9 не получил чанка: {sorted(by_clause)}"
    assert "пожарные клапаны" in by_clause["7.6.9"].text
    assert by_clause["7.6.9"].section == "7"
    # склейка мелких пунктов работает и не теряет номера
    merged = [chunk for chunk in chunks if len(chunk.clauses) > 1]
    assert merged, "ни один чанк не несёт нескольких пунктов — склейка не работает"
    assert all(chunk.clause_id in chunk.clauses for chunk in merged)
    tables = [chunk for chunk in chunks if chunk.kind == "table"]
    assert tables, "таблицы не распознаны как чанки kind=table"
    assert any("Кухня с газовой плитой" in table.text for table in tables)
    # раздел не теряется: каждый чанк либо в разделе, либо во вводной части
    assert all(chunk.section or chunk.section.startswith("Приложение") or chunk.section == "" for chunk in chunks)
    # длинных чанков быть не должно — oversized режется по абзацам
    too_long = [chunk.ord for chunk in chunks if len(chunk.text) > 1800]
    assert not too_long, f"чанки длиннее лимита: {too_long}"


def test_chunks_do_not_cross_section_boundary(fixture_pdf):
    """Мелкие пункты склеиваются только внутри одного раздела."""
    from norms_base.chunk import chunk_markdown

    doc = pdf_to_markdown(fixture_pdf, CODE)
    for chunk in chunk_markdown(doc.text):
        if chunk.clause_id and chunk.clause_id.startswith("1."):
            assert chunk.section == "1", f"пункт {chunk.clause_id} утаскан в раздел {chunk.section}"


# ------------------------------------------------------------- FTS / пункт


def test_fts_finds_exact_wording(ingested):
    hits = ingested.store.search_fts("предел огнестойкости")
    assert hits, "FTS не нашёл 'предел огнестойкости'"
    assert hits[0].clause_id == "7.6.9", f"топ-1 FTS: {hits[0].clause_id}"


def test_clause_lookup_returns_full_text(ingested):
    hits = search.clause(CODE, "7.6.9", store=ingested.store)
    assert len(hits) == 1, f"ожидался один чанк пункта, получено {len(hits)}"
    hit = hits[0]
    assert "пожарные клапаны" in hit.text
    assert hit.page_start == 2
    assert hit.cite.startswith(f"{CODE}, п. 7.6.9")


def test_section_lookup(ingested):
    hits = search.section(CODE, "7", store=ingested.store)
    assert hits, "раздел 7 не найден"
    clause_ids = [hit.clause_id for hit in hits]
    assert "7.6.9" in clause_ids
    assert hits == sorted(hits, key=lambda h: h.chunk_id), "чанки раздела не в порядке документа"


# --------------------------------------------------------------- вектора


def test_merged_clause_is_findable_by_its_own_number(ingested):
    """Пункт, склеенный с соседями, обязан находиться по СВОЕМУ номеру.

    В фикстуре пункты 1.1-1.3 склеены в один чанк с primary-номером 1.1;
    запрос «1.2» должен вернуть этот же чанк, но с корректной ссылкой на 1.2
    и с полным списком соседних пунктов.
    """
    primary = search.clause(CODE, "1.1", store=ingested.store)
    assert primary and len(primary[0].clause_ids) >= 3, "пункты 1.x не склеены в один чанк"

    hits = search.clause(CODE, "1.2", store=ingested.store)
    assert hits, "склеенный пункт 1.2 не найден"
    hit = hits[0]
    assert hit.chunk_id == primary[0].chunk_id, "по номеру 1.2 должен вернуться тот же чанк"
    assert hit.clause_id == "1.2", f"цитата указывает на {hit.clause_id}"
    assert "1.1" in hit.clause_ids and "1.3" in hit.clause_ids, f"соседи потеряны: {hit.clause_ids}"
    assert hit.cite == f"{CODE}, п. 1.2, с. 1", f"неожиданная ссылка: {hit.cite}"
    payload = hit.as_dict()
    assert payload["clause_ids"] == list(hit.clause_ids)


@require_model
def test_vector_search_hits_target_clause(ingested):
    hits = ingested.vector_store.search(
        ingested.embedder.embed_one("предел огнестойкости транзитных воздуховодов"), limit=3
    )
    assert hits, "векторный поиск ничего не вернул"
    top = hits[0].payload
    assert "7.6.9" in (top.get("clause_ids") or []), f"топ-1 вектора: {top.get('clause_ids')}"
    assert top["doc_code"] == CODE
    assert top["page_start"] == 2
    assert "score" not in top  # score идёт из ответа, не из payload


@require_model
def test_hybrid_merges_vector_and_fts(ingested):
    hits = search.hybrid(
        "куда можно выбрасывать воздух из вентиляции",
        k=5,
        store=ingested.store,
        vector_store=ingested.vector_store,
        embedder=ingested.embedder,
    )
    assert hits, "hybrid ничего не вернул"
    top_clauses = [cid for hit in hits[:3] for cid in hit.clause_ids]
    assert "7.6.10" in top_clauses, f"пункт про выброс не в топ-3: {top_clauses}"
    assert any(hit.sources == ["vector", "fts"] or set(hit.sources) == {"vector", "fts"} for hit in hits), (
        f"ни один hit не собран из обоих источников: {[hit.sources for hit in hits]}"
    )


@require_model
def test_semantic_respects_threshold(ingested):
    """Порог должен отсекать заведомо чужие тексты, а целевой — пропускать."""
    target = search.semantic(
        "предел огнестойкости воздуховодов",
        k=3,
        store=ingested.store,
        vector_store=ingested.vector_store,
        embedder=ingested.embedder,
    )
    assert target and "7.6.9" in target[0].clause_ids, f"цель не в топе: {target and target[0].clause_ids}"
    # запрос по совсем другой теме: с большим порогом выдачи не должно остаться
    noise = search.semantic(
        "порядок согласования изысканий для строительства линейных объектов",
        k=3,
        store=ingested.store,
        vector_store=ingested.vector_store,
        embedder=ingested.embedder,
        score_threshold=0.75,
    )
    assert noise == [], f"порог 0.75 пропустил чужой запрос: {[h.clause_id for h in noise]}"


# ------------------------------------------------------------------ статус


def test_status_reports_zero_delta(ingested):
    info = pipeline.status(ingested.store, ingested.vector_store)
    assert info["documents"] == 1
    assert info["chunks"] == ingested.chunk_count
    assert info["embedded"] == info["chunks"]
    assert info["pending_embedding"] == 0
    assert info["qdrant_points"] == info["embedded"]
    assert info["sync_delta"] == 0, f"рассинхрон SQLite <-> Qdrant: {info}"


def test_reingest_is_idempotent(ingested, tmp_path):
    """Повторный add_document не плодит точки и не меняет счётчики."""
    before = _point_ids(ingested)
    info_before = pipeline.status(ingested.store, ingested.vector_store)

    pipeline.add_document(
        CODE,
        "Отопление, вентиляция и кондиционирование воздуха",
        pdf_path=ingested.result.pdf_path,
        store=ingested.store,
        vector_store=ingested.vector_store,
        embedder=ingested.embedder,
        verbose=False,
    )

    after = _point_ids(ingested)
    info_after = pipeline.status(ingested.store, ingested.vector_store)
    assert after == before, f"набор точек изменился: {len(before)} -> {len(after)}"
    assert info_after["sync_delta"] == 0
    assert info_after["chunks"] == info_before["chunks"]
    assert info_after["embedded"] == info_after["chunks"]


def _point_ids(ingested) -> set[str]:
    ids: set[str] = set()
    offset = None
    while True:
        points, offset = ingested.vector_store.client.scroll(
            collection_name=ingested.vector_store.collection,
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        ids.update(str(point.id) for point in points)
        if offset is None:
            return ids
