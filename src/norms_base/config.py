"""Пути и константы базы.

Всё, что меняется между машинами или инстансами, лежит здесь — остальные модули
импортируют константы, а не хардкодят пути.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: SQLite с документами, чанками и полнотекстовым индексом FTS5.
DB_PATH = Path(os.environ.get("NORMS_DB", REPO_ROOT / "data" / "normbase.db"))

#: Скачанные PDF и сконвертированный Markdown (артефакты конверсии, не источник правды).
DATA_DIR = DB_PATH.parent
PDF_DIR = DATA_DIR / "pdf"
MD_DIR = DATA_DIR / "md"

#: Вырезанные из PDF картинки (файлы таблицы figures, data/img/<код>).
IMG_DIR = DATA_DIR / "img"

#: Конфиг описания картинок: vision.model пустой — описывает агент.
FIGURES_YAML = REPO_ROOT / "figures.yaml"

#: Векторный индекс: 1024d Cosine — то же пространство, что у ws-* коллекций и memory-store.
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.environ.get("NORMS_COLLECTION", "norm_docs")
DIM = 1024

#: Локальный ONNX-экспорт BAAI/bge-m3 (тот же, что у memory-store MCP).
BGE_M3_DIR = Path(
    os.environ.get(
        "NORMS_BGE_M3",
        r"C:\Users\olegs\AppData\Local\hermes\models\bge-m3-onnx",
    )
)

#: Целевой размер чанка в символах: меньше ~600 — выдача мелкая, больше ~1800 —
#: вектор размывается и теряет точность на узких пунктах нормы.
CHUNK_MIN_CHARS = 600
CHUNK_MAX_CHARS = 1800

#: Страховка по токенам: чанки реально в 3-4 раза короче, но обрезать хвост
#: молча нельзя — длинный вход не должен упираться в лимит позиций модели.
MAX_TOKENS = 2048

#: UUID5-нейтрон для точек Qdrant: id должен быть детерминированным,
#: иначе переиндексация плодит дубли точек.
POINT_NAMESPACE = "norms_base.norm_docs"

#: Размер батча эмбеддинга. Замер на СП 60.13330.2020 (2026-10-07, 16 логических
#: ядер): 8 длинных чанков за 18.0 с при intra=8 против 23.5 с при intra=4;
#: батчами длиннее выигрыша нет — все тексты паддятся до самого длинного в группе,
#: и 64 текста по 1800 символов считаются дольше, чем четыре батча по 16.
EMBED_BATCH = 16

#: Потоки onnxruntime. Больше 8 не лучше: intra=16 даёт 29 с против 18 с.
EMBED_THREADS = int(os.environ.get("NORMS_EMBED_THREADS", "8"))


def ensure_dirs() -> None:
    """Создаёт каталоги данных (вызывается CLI перед работой)."""
    for path in (DATA_DIR, PDF_DIR, MD_DIR, IMG_DIR):
        path.mkdir(parents=True, exist_ok=True)
    if DB_PATH.parent != DATA_DIR:  # pragma: no cover - защита от рассинхрона путей
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
