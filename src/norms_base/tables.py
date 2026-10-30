"""Структурированные таблицы из Markdown-конверсии: ячейки для SQLite.

pymupdf4llm отдаёт таблицы строками ``|…|…|`` — этот модуль разбирает их в
структуру (шапка, строки, ячейки с нормализованными числами), чтобы база была
источником правды для расчётно-проверочных скриптов: скрипт берёт значения
из ``table_cells``, а не парсит текст чанка заново.

Что считается таблицей: непрерывный блок строк, начинающихся с ``|``; границы
шапки — строка-разделитель ``|---|`` (её кладёт сам pymupdf4llm, повторённые
шапки на стыке страниц уже вырезаны ``convert.join_pages``). Подпись таблицы —
ближайшая строка над блоком, начинающаяся с «Таблица …» или «Продолжение
таблицы …» (обычно это отдельная строка вне блока).

Числа: в ячейке сохраняется сырое значение (``raw`` — источник правды) и
отдельно разобранное ``num`` (русская запятая, пробелы-разделители тысяч).
Всё, что неоднозначно («2/3», «–», «1/ч»), даёт ``num = NULL`` — расчётный
скрипт получает текст и решает сам, а молча неверное число опаснее.

Границы страниц берутся из карты смещений конверсии (``page_for``), поэтому
таблица всегда отвечает «на какой странице я стою».
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

#: Строка-разделитель Markdown-таблицы: ``|---|``, ``| :-: | ---: |``.
RE_TABLE_SEP = re.compile(r"^\|(\s*:?-+:?\s*\|)+\s*$")

#: Подпись: «Таблица П.1 - …», «Таблица 1», «Продолжение/Окончание таблицы 2»,
#: «Содержание». Нормативы набирают подпись с разрядкой («Т а б л и ц а Л.2») —
#: такие варианты схлопываются ДО проверки (RE_SPACED_CAPTION).
RE_CAPTION = re.compile(
    r"^(?:Таблица\s+[А-ПA-Z0-9][\w.\-–—]*"
    r"|(?:Продолжение|Окончание|Начало)\s+таблицы\s+[А-ПA-Z0-9][\w.\-–—]*"
    r"|Содержание)",
    re.IGNORECASE,
)

#: «Т а б л и ц а» -> «Таблица» (подписи через разрядку букв).
RE_SPACED_CAPTION = re.compile(r"Т\s+а\s+б\s+л\s+и\s+ц\s+а", re.IGNORECASE)

#: Сколько непустых строк вверх искать подпись (подпись бывает в 2-3 строки).
CAPTION_LOOK_BACK = 4

#: Число в ячейке: «0,35», «-15», «1 500», «60» — после нормализации.
RE_NUMBER = re.compile(r"[+-]?\d+(?:\.\d*)?")

#: Разделители тысяч в числах нормативов (пробел и неразрывный пробел).
THOUSANDS = "   "


@dataclass(slots=True)
class TableData:
    """Таблица, найденная в Markdown-тексте документа."""

    ord: int                      #: порядок в документе, с 0
    caption: str                  #: «Таблица П.1 - Нормативный воздухообмен…» или ""
    page_start: int
    page_end: int
    start: int                    #: смещение начала блока в doc.text
    end: int                      #: смещение конца блока (за последним \n)
    header: list[str] | None      #: строки до разделителя; None — таблицы без шапки
    rows: list[list[str]]         #: строки данных (ячейки уже почищены)

    @property
    def col_count(self) -> int:
        widths = [len(self.header or [])] + [len(row) for row in self.rows]
        return max(widths, default=0)


def parse_number(raw: str) -> float | None:
    """Число из ячейки или None, если значение неоднозначно.

    «0,35» -> 0.35, «1 500» -> 1500, «60» -> 60.0, «–»/«2/3»/«1/ч» -> None.
    Точка в конце («15,») допускается — так нормативы пишут целые.
    """
    text = str(raw).strip()
    if not text:
        return None
    for char in THOUSANDS:
        text = text.replace(char, "")
    # тире/дефис без цифр — «значение отсутствует», а не минус
    if not any(ch.isdigit() for ch in text):
        return None
    text = text.replace(",", ".")
    if RE_NUMBER.fullmatch(text):
        try:
            return float(text)
        except ValueError:  # pragma: no cover - fullmatch уже гарантирует формат
            return None
    return None


def extract_tables(
    text: str, page_for: Callable[[int], int]
) -> list[TableData]:
    """Все Markdown-таблицы документа с номерами страниц и подписьями.

    ``page_for(offset) -> page_number`` — карта смещений конверсии
    (``DocMarkdown.page_for``).
    """
    lines = text.split("\n")
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line) + 1

    tables: list[TableData] = []
    index = 0
    total = len(lines)
    while index < total:
        if not lines[index].strip().startswith("|"):
            index += 1
            continue
        start = index
        while index < total and lines[index].strip().startswith("|"):
            index += 1
        header, rows = _parse_block(lines[start:index])
        if not rows and not header:
            continue
        block_start = offsets[start]
        block_end = offsets[index - 1] + len(lines[index - 1])
        tables.append(
            TableData(
                ord=len(tables),
                caption=_caption_before(lines, start),
                page_start=page_for(block_start),
                page_end=page_for(max(block_end - 1, block_start)),
                start=block_start,
                end=block_end,
                header=header,
                rows=rows,
            )
        )
    return tables


def rows_payload(table: TableData) -> dict[str, Any]:
    """Словарь для ``Store.replace_tables`` (плюс ``match_text`` для чанка)."""
    return {
        "ord": table.ord,
        "caption": table.caption,
        "page_start": table.page_start,
        "page_end": table.page_end,
        "header": list(table.header or []),
        "rows": [list(row) for row in table.rows],
        "match_text": match_text(table),
    }


def match_text(table: TableData) -> str:
    """Подстрока для поиска чанка-носителя: самая длинная ячейка первой строки.

    Первая строка (шапка или первая строка данных) уникальна в пределах
    документа в подавляющем большинстве случаев; короткие вхождения вроде
    «Помещение» встречаются и в других таблицах, поэтому берётся самая
    длинная ячейка, а не первая попавшаяся.
    """
    source_rows = table.rows or ([] if not table.header else [table.header])
    for row in source_rows:
        candidates = [cell for cell in row if len(cell.strip()) >= 6]
        if candidates:
            return max(candidates, key=len).strip()
    # таблица целиком из коротких ячеек («Да/Нет») — идём в погрешность
    flat = [cell.strip() for row in source_rows for cell in row if cell.strip()]
    return max(flat, key=len) if flat else ""


def sync_tables(
    store: Any,
    doc_id: int,
    doc_code: str,
    text: str,
    page_for: Callable[[int], int],
    *,
    verbose: bool = True,
) -> int:
    """Пересыпает таблицы документа в SQLite (полная замена, идемпотентно).

    Описаний у таблиц нет (их роль выполняет подпись), поэтому замена ничего
    не теряет; чанк-носитель ищется заново — после переиндексации его id
    меняются.
    """
    tables = extract_tables(text, page_for)
    count = store.replace_tables(doc_id, [rows_payload(table) for table in tables])
    if verbose and tables:
        cells = sum(len(table.rows) for table in tables)
        print(f"[tables] {count} таблиц, {cells} строк данных -> {doc_code}", flush=True)
    return count


# ------------------------------------------------------------------ internal

def _parse_block(block: Sequence[str]) -> tuple[list[str] | None, list[list[str]]]:
    """Разбирает блок ``|``-строк: (шапка, строки данных).

    Без разделителя шапки не бывает — тогда весь блок уходит в данные
    (потерять содержимое хуже, чем сохранить таблицу без шапки).
    """
    sep_index = None
    for index, line in enumerate(block):
        if RE_TABLE_SEP.match(line.strip()):
            sep_index = index
            break
    if sep_index is None:
        rows = [_cells(line) for line in block]
        return None, [row for row in rows if any(cell for cell in row)]
    header = _cells(block[sep_index - 1]) if sep_index > 0 else None
    # строк перед разделителем бывает больше одной (двухъярусная шапка) —
    # они уходят в данные, потерять содержимое хуже, чем запомнить лишнюю строку
    leading = [_cells(line) for line in block[: max(sep_index - 1, 0)]]
    rows = leading + [_cells(line) for line in block[sep_index + 1 :]]
    rows = [row for row in rows if any(cell for cell in row)]
    if header is not None and not any(cell for cell in header):
        header = None
    rows = [row for row in rows if not _is_sep_row(row)]
    return header, rows


RE_SEP_CELL = re.compile(r"^:?-+:?$")


def _is_sep_row(cells: Sequence[str]) -> bool:
    """True — строка-разделитель («---|---»), попавшая в данные."""
    non_empty = [cell for cell in cells if cell]
    return bool(non_empty) and all(RE_SEP_CELL.match(cell) for cell in non_empty)


def _cells(line: str) -> list[str]:
    """``| a | b |`` -> ['a', 'b'] (пустые крайние ячейки отбрасываются)."""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [cell.strip() for cell in body.split("|")]


def _caption_before(lines: Sequence[str], start: int) -> str:
    """Подпись таблицы: до CAPTION_LOOK_BACK непустых строк над блоком.

    Поиск идёт вверх до первой подписи; на строке данных соседней таблицы,
    на заголовке раздела или жирном заголовке — останавливается (чужую
    подпись выше заголовка присваивать нельзя).
    """
    index = start - 1
    seen = 0
    while index >= 0 and seen < CAPTION_LOOK_BACK:
        stripped = lines[index].strip()
        if not stripped:
            index -= 1
            continue
        seen += 1
        if stripped.startswith("|"):
            return ""
        candidate = stripped.strip("#*_ \t")
        candidate = RE_SPACED_CAPTION.sub("Таблица", candidate)
        if RE_CAPTION.match(candidate):
            return candidate[:300]
        if re.match(r"#{1,4}\s", stripped):
            return ""
        if stripped.startswith("**") and stripped.endswith("**"):
            return ""
        index -= 1
    return ""
