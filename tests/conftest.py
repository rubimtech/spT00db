"""Общие фикстуры: тестовый PDF и однократный инжест в тестовую базу/коллекцию.

Инжест в сессионной фикстуре выполняется один раз на весь прогон pytest:
прогрев bge-m3 (~8 с) и конверсия PDF не должны оплачиваться в каждом тесте.
Тестовая коллекция ``norm_docs_test`` удаляется в конце сессии, чтобы не
оставлять мусор в общем инстансе Qdrant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from norms_base import config, pipeline  # noqa: E402
from norms_base.config import BGE_M3_DIR  # noqa: E402
from norms_base.embed import Embedder  # noqa: E402
from norms_base.storage import Store  # noqa: E402
from norms_base.vectorstore import VectorStore  # noqa: E402

CODE = "СП 60.13330.2020"
TITLE = "Отопление, вентиляция и кондиционирование воздуха"
TEST_COLLECTION = "norm_docs_test"

MODEL_OK = (BGE_M3_DIR / "model.onnx").exists()

require_model = pytest.mark.skipif(
    not MODEL_OK,
    reason=f"модель bge-m3 не найдена в {BGE_M3_DIR} (задайте NORMS_BGE_M3)",
)

FONT = Path(r"C:\Windows\Fonts\arial.ttf")


def build_fixture_pdf(path: Path) -> Path:
    """Мини-«СП» на 4 страницы: колонтитул, номера страниц, рваный абзац, таблица.

    Таблица нарочно разорвана границей страницы со повторённой шапкой —
    именно это ломает конвертер, если чистка сделана одной прогонкой.
    """
    import pymupdf

    doc = pymupdf.open()
    pages: list[list[str]] = [
        [
            f"{CODE} Отопление, вентиляция и кондиционирование воздуха",
            "1 Область применения",
            "1.1 Настоящий свод правил распространяется на системы отопления, вентиляции "
            "и кондиционирования воздуха жилых, общественных и административно-бытовых "
            "зданий и устанавливает требования к их проектированию, устройству и эксплуатации.",
            "1.2 Требования настоящего свода правил не распространяются на специальные "
            "производственные помещения, для которых установлены особые параметры микроклимата.",
            "1.3 При проектировании систем вентиляции должно предусматриваться удаление из "
            "воздуха вредных веществ, избыточного тепла и избыточной влаги, а также поддержание "
            "допустимых концентраций вредных веществ в рабочей зоне. Расчет воздухообмена "
            "выполняется по методике, приведенной в приложении П.",
        ],
        [
            "7 Воздухообмен и вентиляция",
            "7.6.9 Предел огнестойкости транзитных воздуховодов, проходящих через "
            "противопожарные преграды, должен быть не ниже предела огнестойкости этой преграды. "
            "В местах прохода через противопожарные преграды устанавливаются пожарные клапаны.",
            "7.6.10 Выброс отработанного воздуха из систем вентиляции должен осуществляться "
            "на высоте не менее 2 м от уровня земли и не менее 1 м над кровлей здания. Выброс "
            "в оконные и дверные проемы не допускается, если расстояние до них менее 8 м.",
            "Продолжение: расчетная температура воздуха в помещении принимается в соответствии "
            "с требованиями нормативных документов по тепловой защите ограждающих конструкций, "
            "а кратность воздухообмена устанавливается",
        ],
        [
            "по назначению помещения и числу постоянно находящихся в нем людей. Минимальный "
            "воздухообмен в жилых помещениях должен составлять не менее 0,35 объема помещения в час.",
            "Приложение П",
            "Таблица П.1 - Нормативный воздухообмен помещений",
        ],
        ["Таблица П.1 (продолжение)"],
    ]
    table_rows = [
        [
            ["Помещение", "Воздухообмен, 1/ч", "Примечание"],
            ["Кухня с газовой плитой", "60", "вытяжка над плитой"],
            ["Ванная", "25", "приток через зазор под дверью"],
            ["Туалет", "50", "вытяжка в вентканал"],
        ],
        [
            ["Помещение", "Воздухообмен, 1/ч", "Примечание"],
            ["Жилая комната", "0,35", "кратность от объема"],
            ["Спальня", "0,35", "кратность от объема"],
            ["Кладовая", "1,5", "однократный обмен"],
        ],
    ]

    # Заголовки рисуются крупнее корпуса — pymupdf4llm опознаёт их как
    # Markdown-заголовок («## 1 Область применения»), как в настоящем СП.
    headings = {
        f"{CODE} Отопление, вентиляция и кондиционирование воздуха",
        "1 Область применения",
        "7 Воздухообмен и вентиляция",
        "Приложение П",
    }

    for index, lines in enumerate(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_font(fontname="arial", fontfile=str(FONT))
        # колонтитул (на 1-й странице с отметкой редакции — два разных вида) и номер страницы
        header = f"{CODE} (проект, вторая редакция)" if index == 0 else CODE
        page.insert_text((50, 36), header, fontname="arial", fontsize=9, fontfile=str(FONT))
        page.insert_text((50, 810), str(index + 1), fontname="arial", fontsize=9, fontfile=str(FONT))
        y = 70
        for text in lines:
            size = 14 if text in headings else 10
            rect = pymupdf.Rect(50, y, 550, y + 130)
            remaining = page.insert_textbox(rect, text, fontname="arial", fontsize=size, align=0)
            if remaining < 0:
                raise AssertionError(f"текст не поместился на странице: {text[:60]!r}")
            y += 130 - remaining + 8
        if index >= 2:
            _draw_table(page, y, table_rows[index - 2])

    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    return path


def _draw_table(page, y: float, rows: list[list[str]]) -> float:
    import pymupdf

    widths = [180, 130, 150]
    row_h = 22.0
    x0 = 50.0
    for row_index, row in enumerate(rows):
        x = x0
        for cell_index, cell in enumerate(row):
            page.insert_textbox(
                pymupdf.Rect(x + 3, y + 4, x + widths[cell_index] - 3, y + row_h - 2),
                cell,
                fontname="arial",
                fontsize=9,
            )
            x += widths[cell_index]
        y += row_h
    total = sum(widths)
    y_top = y - len(rows) * row_h
    for step in range(len(rows) + 1):
        line_y = y_top + step * row_h
        page.draw_line(pymupdf.Point(x0, line_y), pymupdf.Point(x0 + total, line_y), width=0.7)
    for step in range(len(widths) + 1):
        x = x0 + sum(widths[:step])
        page.draw_line(pymupdf.Point(x, y_top), pymupdf.Point(x, y), width=0.7)
    return y + 12


class Ingested:
    """Результат сессионного инжеста: доступ к базе, индексу и метрикам."""

    def __init__(self, store: Store, vector_store: VectorStore, embedder: Embedder, result) -> None:
        self.store = store
        self.vector_store = vector_store
        self.embedder = embedder
        self.result = result

    @property
    def chunk_count(self) -> int:
        row = self.store.conn.execute(
            "SELECT COUNT(*) AS n FROM chunks c JOIN documents d ON d.doc_id = c.doc_id WHERE d.code = ?",
            (CODE,),
        ).fetchone()
        return int(row["n"])


@pytest.fixture(scope="session")
def ingested(tmp_path_factory: pytest.TempPathFactory):
    if not MODEL_OK:
        pytest.skip(f"модель bge-m3 не найдена в {BGE_M3_DIR}")

    work = tmp_path_factory.mktemp("ingest")
    pdf = build_fixture_pdf(work / "fixture_sp.pdf")

    # пути инжеста уводим в tmp, чтобы тесты не трогали data/ рабочей базы
    patch = pytest.MonkeyPatch()
    for module in (config, pipeline):
        # DATA_DIR есть только в config — pipeline наследует пути через ensure_dirs
        patch.setattr(module, "DATA_DIR", work, raising=False)
        patch.setattr(module, "PDF_DIR", work / "pdf", raising=False)
        patch.setattr(module, "MD_DIR", work / "md", raising=False)

    store = Store(work / "test_normbase.db")
    store.init_schema()
    vector_store = VectorStore(collection=TEST_COLLECTION)
    vector_store.ensure_collection(recreate=True)
    embedder = Embedder()
    try:
        result = pipeline.add_document(
            CODE,
            TITLE,
            pdf_path=pdf,
            store=store,
            vector_store=vector_store,
            embedder=embedder,
            verbose=False,
        )
        yield Ingested(store, vector_store, embedder, result)
    finally:
        store.close()
        try:
            vector_store.client.delete_collection(TEST_COLLECTION)
        finally:
            vector_store.close()
        patch.undo()


@pytest.fixture(scope="session")
def fixture_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_fixture_pdf(tmp_path_factory.mktemp("fixture") / "fixture_sp.pdf")
