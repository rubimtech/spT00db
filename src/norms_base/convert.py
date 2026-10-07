"""PDF -> Markdown с сохранением номеров страниц.

Почему построчно, а не одним прогоном pymupdf4llm:

* бегущие колонтитулы, номера страниц и отметки редакции живут ровно на
  «своей» странице — чистить их проще и безопаснее до склейки страниц;
* на границе страниц рвётся абзац (предложение продолжается на следующей)
  и рвётся таблица (шапка повторяется или подменяется колонтитулом) — эти
  два разрыва лечатся только при склейке, когда видны оба соседа.

Правила чистки параметризованы кодом документа: колонтитул узнаётся по коду
(``СП 60.13330.2020``), а не по захардкоженной строке, поэтому модуль
работает для любых СП/ГОСТ/СанПиН из базы.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

#: Колонтитул/отметка редакции: «(проект, вторая редакция)», «проект 2 редакция».
EDITION_TAIL = re.compile(r"^\(?(?:проект[^()]{0,40}редакция|издание[^()]{0,30})\)?$", re.IGNORECASE)

#: Номер страницы: арабские (109 стр.) и римские фронтисписа (II…VI).
PAGE_NUMBER = re.compile(r"^(\d{1,3}|I{1,3}|IV|V|VI)$")

RE_MARKUP = re.compile(r"[#*`<>|]")
RE_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
RE_EDITION_PREFIX = re.compile(
    r"^\s*#{0,3}\s*_?\(?(?:<u>)?проект[^_)\n<]{0,40}редакция\)?(?:</u>)?_?\s*",
    re.IGNORECASE,
)
RE_TABLE_SEP = re.compile(r"^\|(\s*:?-+:?\s*\|)+\s*$")


@dataclass(slots=True)
class PageText:
    number: int
    text: str


@dataclass(slots=True)
class DocMarkdown:
    """Готовый Markdown + карта смещений для ответа «на какой странице это»."""

    text: str
    spans: list[tuple[int, int]] = field(default_factory=list)  # (start_offset, page_number)
    page_count: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def page_for(self, offset: int) -> int:
        if not self.spans:
            return 0
        starts = [start for start, _ in self.spans]
        index = bisect.bisect_right(starts, offset) - 1
        return self.spans[max(index, 0)][1]


def pdf_to_markdown(pdf_path: Path | str, doc_code: str) -> DocMarkdown:
    """Конвертирует PDF в чищенный Markdown с картой страниц."""
    import pymupdf4llm

    raw_pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True, verbose=False)
    if not raw_pages:
        raise ValueError(f"pymupdf4llm не вернул страниц для {pdf_path}")

    meta = dict(raw_pages[0].get("metadata") or {})
    pages = [
        PageText(number=int(item.get("metadata", {}).get("page_number") or index + 1),
                 text=clean_page(str(item.get("text") or ""), doc_code))
        for index, item in enumerate(raw_pages)
    ]
    text, spans = join_pages(pages)
    return DocMarkdown(
        text=text,
        spans=spans,
        page_count=int(meta.get("page_count") or len(pages)),
        meta=meta,
    )


# --------------------------------------------------------------------- clean

def clean_page(text: str, doc_code: str) -> str:
    """Убирает колонтитулы, номера страниц и мусор одной страницы."""
    lines = text.split("\n")
    non_empty = [index for index, line in enumerate(lines) if line.strip()]
    edge = set(non_empty[:2]) | set(non_empty[-2:])

    kept: list[str] = []
    skip_next = False
    for index, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _is_header_like(stripped, doc_code):
            continue
        if index in edge and PAGE_NUMBER.match(stripped):
            continue
        if stripped.startswith("|") and doc_code in stripped and _is_header_like(stripped, doc_code):
            # шапка страницы, вплетённая в таблицу: строка + разделитель
            if index + 1 < len(lines) and RE_TABLE_SEP.match(lines[index + 1].strip()):
                skip_next = True
            continue
        prefix_stripped = RE_EDITION_PREFIX.sub("", line, count=1)
        if prefix_stripped != line:
            if prefix_stripped.strip():
                kept.append(prefix_stripped.rstrip())
            continue
        kept.append(line.rstrip())

    return _normalize_blank_lines(kept)


def _is_header_like(line: str, doc_code: str) -> bool:
    """True для строки, состоящей только из кода документа (+ отметки редакции).

    Ссылка на норматив внутри предложения («в соответствии со СП 60.13330.2020…»)
    сюда не попадает — там после кода идёт осмысленный текст.
    """
    bare = RE_MARKUP.sub(" ", RE_BR.sub(" ", line))
    bare = re.sub(r"\s+", " ", bare).strip()
    if bare == doc_code:
        return True
    if bare.startswith(doc_code):
        tail = bare[len(doc_code):].strip()
        return bool(tail) and bool(EDITION_TAIL.match(tail))
    return False


def _normalize_blank_lines(lines: Sequence[str]) -> str:
    """Схлопывает пустые строки и не даёт таблицам прилипать к тексту."""
    out: list[str] = []
    blank = 0
    for line in lines:
        if not line.strip():
            blank += 1
            if blank <= 1:
                out.append("")
            continue
        blank = 0
        is_table = line.strip().startswith("|")
        if out and out[-1].strip() and (is_table != out[-1].strip().startswith("|")):
            out.append("")
        out.append(line)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


# --------------------------------------------------------------------- join

def join_pages(pages: Sequence[PageText]) -> tuple[str, list[tuple[int, int]]]:
    """Склеивает страницы, леча разрывы абзацев и таблиц на границе."""
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    offset = 0
    last_table_header: tuple[str, ...] | None = None

    for page in pages:
        body = page.text.strip("\n")
        if not body.strip():
            continue
        if not parts:
            parts.append(body)
            spans.append((offset, page.number))
            offset += len(body)
            last_table_header = _head_row(body)
            continue

        lines = body.split("\n")
        prev_tail = _last_non_empty(parts[-1])
        next_head = lines[_first_non_empty_index(lines)]
        separator = "\n\n"
        drop_prefix = 0

        if prev_tail.startswith("|") and next_head.startswith("|"):
            repeated = _repeated_header(lines, last_table_header)
            if repeated:
                drop_prefix = repeated
                separator = "\n"
            else:
                separator = "\n"
        elif _needs_continuation(prev_tail, next_head):
            separator = " "

        if drop_prefix:
            body = "\n".join(lines[drop_prefix:])
        parts.append(separator)
        offset += len(separator)
        parts.append(body)
        spans.append((offset, page.number))
        offset += len(body)
        header = _head_row(body)
        if header:
            last_table_header = header

    return "".join(parts), spans


def _needs_continuation(prev_tail: str, next_head: str) -> bool:
    """True, если страница продолжает тот же абзац (рваное предложение)."""
    if prev_tail.startswith("|") or next_head.startswith("|"):
        return False
    if re.search(r"[.!?;:…»\)\]]\s*$", prev_tail) or prev_tail.endswith("-"):
        return False
    return bool(re.match(r"^[а-яёa-z«\"]", next_head.strip()))


def _first_non_empty_index(lines: Sequence[str]) -> int:
    for index, line in enumerate(lines):
        if line.strip():
            return index
    return 0


def _last_non_empty(text: str) -> str:
    for line in reversed(text.split("\n")):
        if line.strip():
            return line
    return ""


def _head_row(body: str) -> tuple[str, ...] | None:
    lines = body.split("\n")
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("|") and not RE_TABLE_SEP.match(stripped):
            return tuple(cell.strip() for cell in stripped.strip("|").split("|"))
        if stripped.startswith("|"):
            continue
        return None
    return None


def _repeated_header(lines: Sequence[str], known: tuple[str, ...] | None) -> int:
    """Сколько строк в начале страницы занято повторённой шапкой таблицы."""
    if not known:
        return 0
    first = lines[_first_non_empty_index(lines)].strip()
    if not first.startswith("|"):
        return 0
    start = _first_non_empty_index(lines)
    header = tuple(cell.strip() for cell in first.strip("|").split("|"))
    if header != known:
        return 0
    if start + 1 < len(lines) and RE_TABLE_SEP.match(lines[start + 1].strip()):
        return start + 2
    return 0
