"""Тесты структурированных таблиц и формул: извлечение -> база -> метка в чанке.

Таблицы проверяются на числах фикстуры (60, 25, 0,35) и на привязке к чанку
``kind=table``; формулы — на детекции тега (А.6) по кропу, записи в базу,
вшитой метке ``<!-- fml:N -->`` (FTS её находит, вектор помечен устаревшим)
и идемпотентной повторной синхронизации.
"""

from __future__ import annotations

import pytest
from conftest import CODE
from norms_base.convert import pdf_to_markdown
from norms_base.formulas import extract_formulas, patch_document_md, patch_markdown, sync_formulas
from norms_base.tables import extract_tables, parse_number

# ------------------------------------------------------------------ числа


def test_parse_number():
    assert parse_number("0,35") == 0.35
    assert parse_number("60") == 60.0
    assert parse_number("1 500") == 1500.0
    assert parse_number("-15") == -15.0
    assert parse_number("15,") == 15.0
    assert parse_number("–") is None          # тире = нет значения
    assert parse_number("2/3") is None        # неоднозначно для скрипта
    assert parse_number("1/ч") is None
    assert parse_number("до 5") is None
    assert parse_number("") is None


# ---------------------------------------------------------------- таблицы


def test_extract_tables_from_fixture(fixture_pdf):
    doc = pdf_to_markdown(fixture_pdf, CODE)
    tables = extract_tables(doc.text, doc.page_for)
    assert len(tables) >= 2, f"таблиц найдено: {len(tables)}"

    first = tables[0]
    assert first.header == ["Помещение", "Воздухообмен, 1/ч", "Примечание"]
    assert first.page_start == 3 and first.page_end == 3
    assert first.caption.startswith("Таблица П.1"), f"подпись: {first.caption!r}"
    assert any(row[0] == "Кухня с газовой плитой" for row in first.rows)
    nums = [parse_number(cell) for row in first.rows for cell in row]
    assert 60.0 in nums and 25.0 in nums, f"числа первой таблицы: {nums}"

    second = tables[1]
    assert second.page_start == 4
    assert second.rows, "продолжение таблицы без строк"
    nums2 = [parse_number(cell) for row in second.rows for cell in row]
    assert 0.35 in nums2, f"числа продолжения: {nums2}"


def test_tables_in_db_bound_to_table_chunk(ingested):
    rows = ingested.store.list_tables(CODE)
    assert rows, "таблицы не попали в базу при инжесте"

    first = next(
        (r for r in rows if r["page_start"] == 3 and r["caption"].startswith("Таблица П.1")),
        None,
    )
    assert first, f"таблица П.1 (стр. 3) не найдена: {[(r['table_id'], r['caption']) for r in rows]}"
    assert first["chunk_id"], "у таблицы нет чанка-носителя"
    chunk = ingested.store.conn.execute(
        "SELECT kind FROM chunks WHERE chunk_id = ?", (first["chunk_id"],)
    ).fetchone()
    assert chunk["kind"] == "table", "таблица привязана не к чанку kind=table"

    table = ingested.store.get_table(first["table_id"])
    assert table["row_count"] == 3 and table["col_count"] == 3
    assert table["header"][0] == "Помещение"
    flat = [cell["num"] for cells in table["rows"] for cell in cells]
    assert 60.0 in flat and 25.0 in flat, f"num-ячейки: {flat}"
    # сырое значение рядом с числом (источник правды не переписан)
    kitchen = table["rows"][0]
    assert kitchen[1] == {"raw": "60", "num": 60.0}

    second = next((r for r in rows if r["page_start"] == 4), None)
    assert second, "продолжение таблицы (стр. 4) не в базе"
    table2 = ingested.store.get_table(second["table_id"])
    flat2 = [cell["num"] for cells in table2["rows"] for cell in cells]
    assert 0.35 in flat2, f"num-ячейки продолжения: {flat2}"


# ---------------------------------------------------------------- формулы


def test_extract_formulas_from_fixture(fixture_pdf, tmp_path):
    found = extract_formulas(fixture_pdf, CODE, img_dir=tmp_path)
    assert found, "формула (А.6) не найдена"
    formula = next((item for item in found if item.tag == "А.6"), None)
    assert formula, f"номера среди кропов: {[item.tag for item in found]}"
    assert formula.page == 3
    assert formula.path.exists() and formula.path.stat().st_size > 500
    # регион содержит и дробь, и тег: тег справа от черты
    x0, y0, x1, y1 = formula.bbox
    assert y0 < 640 < y1, f"чёрта дроби (y=642) вне кропа: {formula.bbox}"


def test_formula_in_db_and_note_survives_reingest(ingested):
    store = ingested.store
    rows = store.list_formulas(CODE)
    assert rows, "формулы не попали в базу при инжесте"
    formula = next((r for r in rows if r["tag"] == "А.6"), None)
    assert formula, f"тега (А.6) среди формул нет: {[r['tag'] for r in rows]}"
    assert formula["description"] is None
    pending_ids = [r["formula_id"] for r in store.list_formulas(CODE, pending=True)]
    assert formula["formula_id"] in pending_ids, "pending-фильтр не видит формулу"

    # описание + выражение: вшиваются в чанк, вектор помечается устаревшим
    row, chunk_id = store.set_formula_data(
        formula["formula_id"],
        expr="353 / (273 + t_h)",
        description="Плотность приточного воздуха при расчетной температуре",
        params={"t_h": "температура наружного воздуха, °С"},
        by="test",
    )
    assert chunk_id, "чанк-носитель формулы не найден"
    assert formula not in store.list_formulas(CODE, pending=True), "описанная формула в pending"
    chunk = store.conn.execute(
        "SELECT text, embed_text, embedded_at FROM chunks WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    marker = f"<!-- fml:{formula['formula_id']} -->"
    assert marker in chunk["text"], "метка не вшита в текст чанка"
    assert "353 / (273 + t_h)" in chunk["embed_text"], "выражение не попало в embed_text"
    assert chunk["embedded_at"] is None, "чанк не помечен 'без вектора'"

    # описание находится FTS вместе с чанком
    hits = store.search_fts("плотность приточного воздуха")
    assert any(hit.chunk_id == chunk_id for hit in hits), "описание формулы не находится FTS"

    # повторная синхронизация: число формул то же, описание сохранено, метка одна
    sync_formulas(store, int(row["doc_id"]), ingested.result.pdf_path, CODE, verbose=False)
    rows2 = store.list_formulas(CODE)
    assert len(rows2) == len(rows), f"дубли формул: {len(rows)} -> {len(rows2)}"
    again = store.get_formula(formula["formula_id"])
    assert again["description"].startswith("Плотность приточного воздуха")
    chunk2 = store.conn.execute(
        "SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,)
    ).fetchone()
    assert chunk2["text"].count(marker) == 1, "метка задублировалась"


def test_formula_note_marks_vectors_stale_and_reembed_fixes(ingested):
    """Описание формулы оставляет чанк без вектора; reembed возвращает СИНХРОНО."""
    from norms_base import pipeline

    store = ingested.store
    stale = store.chunks_missing_embedding()
    assert stale, "после описания формулы чанк не помечен 'без вектора'"
    done = pipeline.reembed_pending(
        store, vector_store=ingested.vector_store,
        embedder=ingested.embedder, verbose=False,
    )
    assert done >= 1, "reembed ничего не доэмбеджил"
    info = pipeline.status(store, ingested.vector_store)
    assert info["pending_embedding"] == 0, info
    assert info["sync_delta"] == 0, f"после reembed расхождение: {info}"


# ------------------------------------------------------- формулы в Markdown


def test_patch_markdown_puts_latex_before_tag(tmp_path):
    md = "по формуле:\n\n(А.6)\n\nгде t — температура"
    rows = [{
        "formula_id": 7, "tag": "А.6", "path": str(tmp_path / "img" / "f.png"),
        "latex": r"\rho = \frac{353}{273 + t_h}", "expr": "",
        "description": "Плотность приточного воздуха",
    }]
    text, inserted = patch_markdown(md, rows, tmp_path)
    assert inserted == 1
    assert "$$" in text and r"\frac{353}{273 + t_h}" in text
    assert "(А.6)" in text, "строка-номер потеряна — чанк-носитель её ищет"
    assert text.index(r"\frac{353}") < text.index("(А.6)")

    again, inserted2 = patch_markdown(text, rows, tmp_path)
    assert inserted2 == 1 and again == text, "повторная вставка задублировала блок"


def test_patch_markdown_without_recognition_keeps_crop_image(tmp_path):
    img = tmp_path / "img" / "f_p003_01.png"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"png")
    rows = [{
        "formula_id": 1, "tag": None, "path": str(img),
        "latex": "", "expr": "", "description": "",
    }]
    text, inserted = patch_markdown("текст документа\n", rows, tmp_path)
    assert inserted == 1
    assert "![Формула без номера](img/f_p003_01.png)" in text
    assert "## Формулы, извлечённые из PDF" in text


def test_patch_markdown_places_by_anchor_when_tag_line_lost(tmp_path):
    """Тело формулы и её номер потеряны конвертером — вставка идёт по якорю."""
    md = (
        "5.2 Нормируемое значение приведенного сопротивления теплопередаче\n"
        "ограждающей конструкции следует определять по формуле\n"
        "\n"
        "где R — базовое значение требуемого сопротивления теплопередаче;\n"
    )
    img = tmp_path / "img" / "f_p011_01.png"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"png")
    rows = [{
        "formula_id": 53, "tag": "5.2", "path": str(img),
        "latex": "", "expr": "", "description": "",
        "anchor": "5.2 Нормируемое значение приведенного сопротивления теплопередаче "
                  "ограждающей конструкции следует определять по формуле",
    }]
    text, inserted = patch_markdown(md, rows, tmp_path)
    assert inserted == 1
    assert "f_p011_01.png" in text
    assert text.index("f_p011_01.png") < text.index("где R — базовое")


def test_patch_document_md_rewrites_saved_file(tmp_path):
    from norms_base.storage import Store

    store = Store(tmp_path / "t.db")
    store.init_schema()
    md = tmp_path / "doc.md"
    md.write_text("по формуле:\n\n(А.6)\n", encoding="utf-8")
    img = tmp_path / "img" / "f_p003_01.png"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"png")
    doc_id = store.upsert_document(code=CODE, title="Т", md_path=str(md))
    store.upsert_formula(
        doc_id=doc_id, tag="А.6", page=3, bbox="0,0,1,1", path=str(img), anchor="",
    )
    formula = next(r for r in store.list_formulas(CODE) if r["tag"] == "А.6")
    store.set_formula_data(
        formula["formula_id"], latex=r"\rho = \frac{353}{273 + t_h}",
        description="Плотность приточного воздуха", by="test",
    )
    inserted = patch_document_md(store, CODE, verbose=False)
    store.close()

    assert inserted == 1
    assert r"\frac{353}{273 + t_h}" in md.read_text(encoding="utf-8")


def test_ingest_writes_formula_blocks_into_md(ingested):
    """Даже без распознавания формула не теряется: в Markdown стоит кроп."""
    md = ingested.result.md_path.read_text(encoding="utf-8")
    assert "<!-- fml-md:А.6 -->" in md, "блок формулы не попал в Markdown при инжесте"
    assert "f_p003_01.png" in md, "в блоке нет ссылки на кроп"
