"""Правки: документы-изменения к своду правил.

* чанкер обязан привязывать «Пункт N.N.N. …» к номеру пункта основного документа;
* колонтитулы официальных изменений (дата, «Изм. N к СП …») вычищаются;
* ``clause()`` по базовому коду подмешивает чанки-правки ``'… (изм. N)'``.
"""

from __future__ import annotations

from conftest import CODE
from norms_base import search
from norms_base.chunk import chunk_markdown
from norms_base.convert import clean_page
from norms_base.storage import Store


AMEND_TEXT = """# **ИЗМЕНЕНИЕ № 4**

**к СП 60.13330.2020 \"СНиП 41-01-2003\"** ОКС 91.140.10

**6.2 Системы отопления**

Пункт 6.2.6. Заменить ссылку: \"СП 50.13330.2016\" на \"СП 50.13330.2024\".

Пункт 6.2.17. Второй абзац. Изложить в новой редакции:

\"В системах водяного отопления для выпуска воздуха следует предусматривать
проточные воздухосборники или краны.\".

Приложение А **Расчет тепловых нагрузок**

Пункт А.2. Формула (А.4). Экспликация. Заменить ссылку.
"""


def test_chunker_binds_pt_form_clause_numbers():
    """«Пункт 6.2.17. …» — граница чанка с номером пункта 6.2.17."""
    chunks = chunk_markdown(AMEND_TEXT)
    clauses = {cid for chunk in chunks for cid in chunk.clauses}
    assert {"6.2.6", "6.2.17", "А.2"} <= clauses, f"номера не привязаны: {sorted(clauses)}"
    by_clause = {cid: chunk for chunk in chunks for cid in chunk.clauses}
    assert "Изложить в новой редакции" in by_clause["6.2.17"].text


def test_chunker_keeps_inline_pt_clause():
    """Номер в склеенной строке (конвертер сшивает абзацы) тоже попадает в чанк."""
    text = "Третий абзац. Заменить слова: \"A\" на \"B\". Пункт 5.8. Четвертый абзац. Исключить."
    chunks = chunk_markdown(text)
    clauses = {cid for chunk in chunks for cid in chunk.clauses}
    assert "5.8" in clauses, f"склеенный пункт потерян: {sorted(clauses)}"


def test_clean_page_drops_amendment_headers():
    """Шапки aeropro (дата + «Изм. N к КОД» + номер) не должны оставаться в тексте."""
    page = "\n".join(
        [
            "01.11.2024",
            "Изм. 3 к СП 60.13330.2020",
            "1",
            "1 Область применения",
            "Пункт 1.2. Заменить слова: \"а\" на \"б\".",
            "21.07.2026",
        ]
    )
    cleaned = clean_page(page, "СП 60.13330.2020 (изм. 3)")
    assert "01.11.2024" not in cleaned
    assert "21.07.2026" not in cleaned
    assert "Изм. 3 к СП 60.13330.2020" not in cleaned
    assert "1 Область применения" in cleaned
    assert "Пункт 1.2." in cleaned


def test_clean_page_keeps_title_but_drops_page_number():
    """Настоящий титул изменения (не колонтитул) обязан сохраниться."""
    page = "\n".join(
        [
            "6 Изменение к СП 60.13330.2020 \"СНиП 41-01-2003\"",
            "2",
            "# **ИЗМЕНЕНИЕ № 6**",
            "Пункт 7.3.6. Заменить слова.",
        ]
    )
    cleaned = clean_page(page, "СП 60.13330.2020 (изм. 6)")
    assert "ИЗМЕНЕНИЕ № 6" in cleaned, "титул изменения вычищен ошибочно"
    assert "Пункт 7.3.6." in cleaned


def _ingest_two_docs(tmp_path) -> Store:
    """Базовый документ + правка в одной тестовой базе (без векторов)."""
    store = Store(tmp_path / "amend.db")
    store.init_schema()
    base_id = store.upsert_document(code=CODE, title="Базовый")
    store.replace_chunks(
        base_id,
        CODE,
        [
            {
                "ord": 0,
                "kind": "text",
                "clause_id": "7.6.9",
                "clause_ids": ["7.6.9"],
                "section": "7",
                "section_title": "Выбросы",
                "heading": "",
                "page_start": 2,
                "page_end": 2,
                "text": "Предел огнестойкости транзитных воздуховодов должен быть не ниже предела огнестойкости преграды.",
                "embed_text": "",
            }
        ],
    )
    am_id = store.upsert_document(code=f"{CODE} (изм. 4)", title="Правка")
    store.replace_chunks(
        am_id,
        f"{CODE} (изм. 4)",
        [
            {
                "ord": 0,
                "kind": "text",
                "clause_id": "7.6.9",
                "clause_ids": ["7.6.9"],
                "section": "",
                "section_title": "",
                "heading": "",
                "page_start": 1,
                "page_end": 1,
                "text": "Пункт 7.6.9. Изложить в новой редакции: «Новый текст пункта».",
                "embed_text": "",
            }
        ],
    )
    return store


def test_clause_mixed_with_amendments(tmp_path):
    """Точная ссылка по базовому коду отдаёт и оригинал, и правку — сначала оригинал."""
    store = _ingest_two_docs(tmp_path)
    try:
        hits = search.clause(CODE, "7.6.9", store=store)
        assert len(hits) == 2, f"ожидались оригинал + правка, получено {len(hits)}"
        assert hits[0].doc_code == CODE, f"сначала должна идти база: {hits[0].doc_code}"
        assert hits[1].doc_code == f"{CODE} (изм. 4)"
        assert "Новый текст пункта" in hits[1].text
        # чужие документы не подмешиваются
        other = store.upsert_document(code="СП 30.13330.2020", title="Чужой")
        store.replace_chunks(
            other,
            "СП 30.13330.2020",
            [
                {
                    "ord": 0,
                    "kind": "text",
                    "clause_id": "7.6.9",
                    "clause_ids": ["7.6.9"],
                    "section": "",
                    "section_title": "",
                    "heading": "",
                    "page_start": 1,
                    "page_end": 1,
                    "text": "Чужой пункт с тем же номером.",
                    "embed_text": "",
                }
            ],
        )
        hits = search.clause(CODE, "7.6.9", store=store)
        assert len(hits) == 2, "чужой документ подмешан по номеру пункта"
    finally:
        store.close()


def test_unrelated_code_not_matched_by_prefix(tmp_path):
    """«СП 60.13330.2020» не должен подхватывать «СП 60.13330.2020-1» без скобки."""
    store = _ingest_two_docs(tmp_path)
    try:
        other = store.upsert_document(code=f"{CODE}-1", title="Другой свод")
        store.replace_chunks(
            other,
            f"{CODE}-1",
            [
                {
                    "ord": 0,
                    "kind": "text",
                    "clause_id": "7.6.9",
                    "clause_ids": ["7.6.9"],
                    "section": "",
                    "section_title": "",
                    "heading": "",
                    "page_start": 1,
                    "page_end": 1,
                    "text": "Не правка, а отдельный документ с похожим кодом.",
                    "embed_text": "",
                }
            ],
        )
        hits = search.clause(CODE, "7.6.9", store=store)
        codes = {hit.doc_code for hit in hits}
        assert f"{CODE}-1" not in codes, f"префикс-ложное срабатывание: {codes}"
    finally:
        store.close()
