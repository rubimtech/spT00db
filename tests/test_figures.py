"""Тесты картинок: извлечение из фикстуры, описание, вшивание в чанк, конфиг LLM.

Фикстурный PDF несёт две картинки: векторную схему с подписью на 1-й странице
(crop) и растровую вставку с подписью на 4-й (raster). Описание в тестах
пишется напрямую в store — сеть и LLM здесь не участвуют.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import CODE
from norms_base import pipeline
from norms_base.figures import extract_figures, load_vision_config


def test_result_counts_figures(ingested):
    assert ingested.result.figures == 2, f"ожидалось 2 картинки, вырезано: {ingested.result.figures}"


def test_extract_finds_raster_and_crop(ingested):
    rows = ingested.store.list_figures(CODE)
    kinds = {row["kind"] for row in rows}
    assert {"raster", "crop"} <= kinds, f"виды картинок: {kinds}"
    for row in rows:
        path = Path(row["path"])
        assert path.exists() and path.stat().st_size > 0, f"файл не сохранён: {path}"
        assert row["page"] >= 1
        assert row["caption"].startswith("Рисунок"), f"подпись не распознана: {row['caption']!r}"
        assert row["description"] is None, "описание появилось раньше, чем его писали"


def test_extract_is_idempotent(fixture_pdf, tmp_path):
    first = extract_figures(fixture_pdf, "Тест", tmp_path)
    second = extract_figures(fixture_pdf, "Тест", tmp_path)
    assert len(first) == len(second) == 2
    assert sorted(fig.path.name for fig in first) == sorted(fig.path.name for fig in second)


def test_describe_writes_note_into_chunk_and_fts(ingested):
    store = ingested.store
    figure = next(row for row in store.list_figures(CODE) if row["kind"] == "crop")
    description = "Теплоноситель остывает экспоненциально, температура падает к концу периода."

    _row, chunk_id = store.set_figure_description(figure["figure_id"], description, by="test")
    assert chunk_id is not None, "чанк-носитель не найден"

    row = store.get_figure(figure["figure_id"])
    assert row["description"] == description
    assert row["described_by"] == "test" and row["described_at"]

    text = store.conn.execute("SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()["text"]
    assert f"<!-- fig:{figure['figure_id']} -->" in text
    assert description in text

    hits = store.search_fts("остывает экспоненциально")
    assert hits and hits[0].chunk_id == chunk_id, "описание не нашлось в FTS"

    # идемпотентно: повторная вшивка не плодит вторую метку
    store.reapply_figure_notes(store.get_document(CODE)["doc_id"])
    store.commit()
    text = store.conn.execute("SELECT text FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()["text"]
    assert text.count(f"<!-- fig:{figure['figure_id']} -->") == 1

    # чанк помечен «переэмбедить» — дособираем, чтобы status остался СИНХРОНО
    pending = store.chunks_missing_embedding()
    if pending:
        pipeline._embed_and_upload(
            pending,
            engine=ingested.embedder,
            store=store,
            vector_store=ingested.vector_store,
            batch_size=16,
            verbose=False,
        )
    assert store.stats()["pending_embedding"] == 0


def test_description_survives_reingest(ingested):
    """Описания хранятся в figures и вшиваются заново после переиндексации."""
    store = ingested.store
    figure = next(row for row in store.list_figures(CODE) if row["description"])
    pipeline.add_document(
        CODE,
        "Отопление, вентиляция и кондиционирование воздуха",
        pdf_path=ingested.result.pdf_path,
        store=store,
        vector_store=ingested.vector_store,
        embedder=ingested.embedder,
        verbose=False,
    )

    row = store.get_figure(figure["figure_id"])
    assert row["description"] == figure["description"], "описание потерялось при re-ingest"

    marker = f"<!-- fig:{figure['figure_id']} -->"
    notes = store.conn.execute(
        "SELECT chunk_id FROM chunks WHERE text LIKE ?", (f"%{marker}%",)
    ).fetchall()
    assert len(notes) == 1, f"метка описания в чанках: {len(notes)} шт. (нужна ровно 1)"
    assert store.stats()["pending_embedding"] == 0


# ------------------------------------------------------------- конфиг LLM


def test_vision_config_agent_mode_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("NORMS_VISION_MODEL", raising=False)
    cfg_file = tmp_path / "figures.yaml"
    cfg_file.write_text('vision:\n  model: ""\n', encoding="utf-8")
    cfg = load_vision_config(cfg_file)
    assert cfg.model == "" and not cfg.auto, "пустая модель должна давать режим агента"


def test_vision_config_resolves_hermes_provider(tmp_path, monkeypatch):
    monkeypatch.delenv("NORMS_VISION_MODEL", raising=False)
    monkeypatch.delenv("NORMS_VISION_API_KEY", raising=False)
    cfg_file = tmp_path / "figures.yaml"
    cfg_file.write_text('vision:\n  model: "test-model"\n  provider: "plusvibeapi"\n', encoding="utf-8")
    cfg = load_vision_config(cfg_file)
    if not cfg.base_url:
        pytest.skip("Hermes config.yaml недоступен")
    assert cfg.base_url.startswith("http")
    assert "plusvibeapi" in cfg.base_url, (
        f"base_url провайдера подмешан aux-vision: {cfg.base_url}"
    )
    if not cfg.api_key:
        pytest.skip("ключ провайдера не в окружении")
    assert cfg.auto, "после подстановки провайдера автоописание должно включиться"
