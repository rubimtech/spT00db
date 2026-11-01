"""Структурный чанкер: режет Markdown по пунктам норматива, а не по символам.

Для проверки проектки важна привязка к пункту («7.6.9», «приложение В»), поэтому
границы чанков — это границы пунктов и таблиц, а не произвольное окно.

* граница нового чанка: заголовок раздела, номер пункта в начале строки,
  начало таблицы;
* мелкие соседние пункты склеиваются до ``CHUNK_MIN_CHARS`` в пределах одного
  раздела (вектор из 150 символов шумит), но никогда через границу раздела —
  иначе теряется контекст «о чём это»;
* таблицы не режутся: строка таблицы вне шапки не читается сама по себе.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .config import CHUNK_MAX_CHARS, CHUNK_MIN_CHARS

RE_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
RE_APPENDIX = re.compile(r"^\*{0,2}Приложение\s+([А-ПA-Z])\.?\*{0,2}\s*$")
#: Номер пункта: цифровой («7.6.9», «1 Область применения») — он же граница раздела.
RE_CLAUSE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+\S")
#: Любой номер пункта, включая пункты приложений («А.1», «Б.2.3», «П.1»):
#: ссылки на них есть в тексте («п. А.2.4 приложения А»), поэтому такие строки
#: обязаны быть границей чанка и попадать в поиск по номеру.
RE_CLAUSE_ANY = re.compile(
    r"^((?:\d{1,2}(?:\.\d{1,2}){0,2})|(?:[А-П]\.\d{1,2}(?:\.\d{1,2})*))\.?\s+\S"
)
#: Пункт, набранный словом (официальные изменения: «Пункт 6.2.17. Изложить…»).
#: Без этой формы чанки изменений не привязываются к номерам пунктов основного
#: документа — точный запрос clause() их не находит.
RE_PT_CLAUSE = re.compile(
    r"^\*{0,2}Пункт[ы]?\s+"
    r"((?:\d{1,2}(?:\.\d{1,2}){0,2})|(?:[А-П]\.\d{1,2}(?:\.\d{1,2})*))\.?\*{0,2}\.?\s+\S"
)
#: То же без привязки к началу строки: в изменениях соседние пункты склеиваются
#: конвертером в одну строку («…подготовка». Пункт 5.8. Четвёртый абзац…») —
#: номер обязан попасть в chunk_clauses, даже если границей чанка он не стал.
RE_PT_INLINE = re.compile(
    r"Пункт[ы]?\s+((?:\d{1,2}(?:\.\d{1,2}){0,2})|(?:[А-П]\.\d{1,2}(?:\.\d{1,2})*))"
)
RE_TABLE = re.compile(r"^\|")
RE_TABLE_SEP = re.compile(r"^\|(\s*:?-+:?\s*\|)+\s*$")
RE_SENTENCE = re.compile(r"(?<=[.!?;])\s+")


@dataclass(slots=True)
class Chunk:
    ord: int
    kind: str            # text | table
    clause_id: str | None
    section: str         # «7.6», «Приложение В», «» для вводной части
    section_title: str
    heading: str         # ближайший заголовок (для контекста эмбеддинга)
    text: str
    start: int
    end: int
    #: все пункты, попавшие в чанк (склейка мелких пунктов не должна терять
    #: их номера — иначе «п. 7.6.9» не находится в склеенном чанке 7.6.8)
    clauses: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return len(self.text)


@dataclass(slots=True)
class _Block:
    kind: str
    clause_id: str | None
    section: str
    section_title: str
    heading: str
    lines: list[str]
    start: int
    end: int
    clauses: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip("\n")

    @property
    def size(self) -> int:
        """Длина текста блока — та же метрика, что у :attr:`Chunk.size`.

        Свойства нет на ``_Block``, ломается весь чанкер: ``_merge_small`` и
        ``_oversized`` меряют блоки именно в символах, а не в строках.
        """
        return len(self.text)


def chunk_markdown(
    text: str,
    *,
    min_chars: int = CHUNK_MIN_CHARS,
    max_chars: int = CHUNK_MAX_CHARS,
) -> list[Chunk]:
    """Режет Markdown на чанки с привязкой к пунктам и смещениями в исходном тексте."""
    blocks = _split_blocks(text, min_chars, max_chars)
    return [
        Chunk(
            ord=index,
            kind=block.kind,
            clause_id=block.clause_id,
            section=block.section,
            section_title=block.section_title,
            heading=block.heading,
            text=block.text,
            start=block.start,
            end=block.end,
            clauses=block.clauses,
        )
        for index, block in enumerate(blocks)
    ]


def build_embed_text(chunk: Chunk, doc_code: str, doc_title: str) -> str:
    """Контекст для вектора: без него запрос по другой теме ловит чужой пункт.

    Шапка содержит код, название и раздел — ровно то, чего нет в теле пункта,
    но что отличает «выброс в атмосферу» в СП по вентиляции от СП по пожарной
    безопасности.
    """
    context = [doc_code, doc_title]
    if chunk.section:
        label = f"Раздел {chunk.section}"
        if chunk.section_title and chunk.section_title != chunk.section:
            label = f"{label} {chunk.section_title}"
        context.append(label)
    elif chunk.section_title:
        context.append(chunk.section_title)
    if chunk.heading and chunk.heading not in context[-1]:
        context.append(chunk.heading)
    head = ". ".join(part for part in context if part)
    return f"{head}. {chunk.text}".strip()


# ------------------------------------------------------------------ internal

def _split_blocks(text: str, min_chars: int, max_chars: int) -> list[_Block]:
    blocks: list[_Block] = []
    offset = 0
    section = ""
    section_title = ""
    pending_heading = ""
    current: _Block | None = None

    def flush() -> None:
        nonlocal current
        # end здесь НЕ пересчитывается: offset уже ушёл за строку-границу
        # (заголовок или новый пункт), и блок захватывал бы чужую строку —
        # страница чанка из-за этого уезжала на страницу дальше.
        if current is not None and current.lines:
            blocks.append(current)
        current = None

    for raw_line in text.split("\n"):
        line_start = offset
        offset += len(raw_line) + 1
        stripped = raw_line.strip()

        if not stripped:
            if current is not None:
                current.lines.append("")
            continue

        heading = _heading_of(stripped)
        if heading is not None:
            flush()
            new_section, new_title = _section_of(heading)
            if new_section:
                section, section_title = new_section, new_title
            else:
                # Ненумерованный заголовок — подзаголовок текущего раздела, а не
                # его конец: заголовок приложения «Расчет тепловых нагрузок…»
                # иначе сбрасывал section в «», и всё приложение теряло привязку.
                section_title = heading
            pending_heading = heading
            continue

        if RE_TABLE.match(stripped):
            if current is not None and current.kind != "table":
                flush()
            if current is None:
                current = _new_block("table", line_start, section, section_title, pending_heading)
                pending_heading = ""
            current.lines.append(raw_line)
            current.end = offset
            continue

        clause_match = RE_CLAUSE_ANY.match(stripped) or RE_PT_CLAUSE.match(stripped)
        starts_clause = clause_match is not None
        clause_id = clause_match.group(1) if clause_match else None
        if starts_clause:
            if current is None or current.clause_id != clause_id or current.kind == "table":
                flush()
                current = _new_block("text", line_start, section, section_title, pending_heading, clause_id)
                pending_heading = ""
        elif current is None or current.kind == "table":
            flush()
            current = _new_block("text", line_start, section, section_title, pending_heading)
            pending_heading = ""

        current.lines.append(raw_line)
        current.end = offset

    flush()
    merged = [block for block in _merge_small(blocks, min_chars) if block.text]
    return _attach_clauses(_oversized(merged, max_chars))


def _attach_clauses(blocks: Sequence[_Block]) -> list[_Block]:
    """Проставляет блоку все пункты, чьи строки в него вошли (пост-проход).

    Склейка мелких пунктов оставляет в блоке только номер первого, и тогда
    «п. 7.6.9» не находился бы: пункт жил в чанке, помеченном как 7.6.8.
    Номера пересчитываются по готовым блокам — так одинаково ведут себя и
    склейка, и разрезка oversized-блока, и таблицы.
    """
    for block in blocks:
        found: list[str] = []
        for line in block.lines:
            stripped = line.strip()
            match = RE_CLAUSE_ANY.match(stripped)
            if match and match.group(1) not in found:
                found.append(match.group(1))
            # «Пункт N.N» — начало строки или середина (склейка конвертером)
            for inline in RE_PT_INLINE.finditer(stripped):
                if inline.group(1) not in found:
                    found.append(inline.group(1))
        if found:
            block.clauses = tuple(found)
            block.clause_id = found[0]
        else:
            block.clauses = (block.clause_id,) if block.clause_id else ()
    return list(blocks)


def _new_block(
    kind: str,
    start: int,
    section: str,
    section_title: str,
    heading: str,
    clause_id: str | None = None,
) -> _Block:
    return _Block(
        kind=kind,
        clause_id=clause_id,
        section=section,
        section_title=section_title,
        heading=heading,
        lines=[],
        start=start,
        end=start,
    )


def _heading_of(line: str) -> str | None:
    match = RE_HEADING.match(line)
    if match:
        return re.sub(r"[*_]", "", match.group(2)).strip()
    if RE_APPENDIX.match(line):
        return re.sub(r"[*_]", "", line).strip()
    # жирный «заголовок на всю строку» — так pymupdf4llm отдаёт приложения
    if line.startswith("**") and line.endswith("**") and len(line) < 70:
        inner = line.strip("*").strip()
        if RE_CLAUSE_ANY.match(inner) is None and "\n" not in inner:
            return inner
    return None


def _section_of(heading: str) -> tuple[str, str]:
    clause = RE_CLAUSE.match(heading)
    if clause:
        number = clause.group(1)
        title = heading[clause.end():].strip()
        return number, title or heading
    appendix = RE_APPENDIX.match(heading)
    if appendix:
        return f"Приложение {appendix.group(1)}", heading
    return "", heading


def _merge_small(blocks: list[_Block], min_chars: int) -> list[_Block]:
    """Дотягивает мелкие блоки до min_chars: вперёд в пределах раздела, затем назад.

    Граница раздела — единственное, что склейка не пересекает: защищать её
    дополнительно запретом на склейку первого блока после заголовка нельзя —
    тогда в базу попадают чанки из 4-11 символов («Стр.», «Таблица Б.1»),
    чьи векторы мусорят выдачу.
    """
    forward: list[_Block] = []
    for block in blocks:
        if (
            forward
            and block.size < min_chars
            and forward[-1].section == block.section
            and forward[-1].size + block.size <= min_chars * 2
        ):
            target = forward[-1]
            target.lines.extend([""] + block.lines)
            target.end = block.end
            target.clause_id = target.clause_id or block.clause_id
            if target.kind != block.kind:
                # таблица важнее текста: подпись к таблице, склеенная с ней,
                # должна остаться чанком kind=table, иначе ломается фильтр по таблицам
                target.kind = "table" if "table" in (target.kind, block.kind) else "text"
            continue
        forward.append(block)

    backward: list[_Block] = []
    for block in forward:
        if (
            backward
            and block.size < min_chars
            and backward[-1].section == block.section
            and backward[-1].size + block.size <= min_chars * 2
        ):
            target = backward[-1]
            target.lines.extend([""] + block.lines)
            target.end = block.end
            target.clause_id = target.clause_id or block.clause_id
            continue
        backward.append(block)
    return backward


def _oversized(blocks: Sequence[_Block], max_chars: int) -> list[_Block]:
    """Разрезает блоки больше max_chars (смещения сохраняются).

    Таблицы режутся по СТРОКАМ с повтором шапки в каждой части: строка вне
    шапки не читается сама по себе, а обрезанный молча хвост уехал бы за
    MAX_TOKENS модели (таблица на 4939 символов — это уже не 2048 токенов).
    """
    result: list[_Block] = []
    for block in blocks:
        if block.size <= max_chars:
            result.append(block)
        elif block.kind == "table":
            result.extend(_split_table(block, max_chars))
        else:
            result.extend(_split_block(block, max_chars))
    return result


def _split_table(block: _Block, max_chars: int) -> list[_Block]:
    """Режет таблицу по строкам, повторяя шапку (строки до разделителя ``|---|``) в каждой части.

    Смещения каждой части — позиция её ПЕРВОЙ СТРОКИ ДАННЫХ в исходном блоке:
    страница считается по данным, а не по повторённой шапке.
    """
    lines = block.lines
    head_end = 0
    for index, line in enumerate(lines):
        if RE_TABLE_SEP.match(line.strip()):
            head_end = index + 1
            break
    head_lines = lines[:head_end]
    head_len = len("\n".join(head_lines))
    if not head_lines:
        head_len = 0

    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line) + 1

    # части: (первая строка данных, последняя строка данных, строки части)
    groups: list[tuple[int, int, list[str]]] = []
    current: list[str] = list(head_lines)
    current_len = head_len
    first_data: int | None = None
    last_data = head_end
    for index in range(head_end, len(lines)):
        line = lines[index]
        extra = len(line) + (1 if current_len else 0)
        # резать можно только когда в текущей части уже есть строки данных —
        # иначе получится пустая часть из одной шапки
        if current_len + extra > max_chars and len(current) > head_end:
            groups.append((first_data if first_data is not None else head_end, last_data, current))
            current = list(head_lines) + [line]
            current_len = len("\n".join(current))
            first_data = index
        else:
            if first_data is None:
                first_data = index
            current.append(line)
            current_len += extra
        last_data = index
    groups.append((first_data if first_data is not None else head_end, last_data, current))

    result: list[_Block] = []
    for start_index, end_index, piece_lines in groups:
        start = block.start + offsets[min(start_index, len(offsets) - 1)]
        end = block.start + offsets[min(end_index, len(offsets) - 1)] + len(lines[end_index])
        result.append(
            _Block(
                kind="table",
                clause_id=block.clause_id,
                section=block.section,
                section_title=block.section_title,
                heading=block.heading if start_index <= head_end else "",
                lines=piece_lines,
                start=start,
                end=end,
            )
        )
    return result


def _split_block(block: _Block, max_chars: int) -> list[_Block]:
    """Режет блок по абзацам/предложениям, восстанавливая смещения поиском по оригиналу.

    Смещения нужны для ответа «на какой странице» — дрейф здесь молча врал бы
    бы о странице, поэтому позиция каждого куска ищется в исходном блоке.
    """
    original = "\n".join(block.lines)
    units: list[tuple[str, str | None]] = []
    for paragraph in original.split("\n\n"):
        pieces = [paragraph] if len(paragraph) <= max_chars else _split_long(paragraph, max_chars)
        for index, piece in enumerate(pieces):
            if not units:
                units.append((piece, None))
            elif index == 0:
                units.append((piece, "\n\n"))
            else:
                units.append((piece, " "))

    texts: list[str] = []
    current = ""
    for text, joiner in units:
        if current and len(current) + len(joiner or "") + len(text) > max_chars:
            texts.append(current)
            current, joiner = "", None
        current = text if not current else f"{current}{joiner}{text}"
    if current:
        texts.append(current)

    blocks: list[_Block] = []
    cursor = 0
    for index, piece in enumerate(texts):
        found = original.find(piece, cursor)
        start = block.start + (found if found >= 0 else cursor)
        blocks.append(
            _Block(
                kind=block.kind,
                clause_id=block.clause_id,
                section=block.section,
                section_title=block.section_title,
                heading=block.heading if index == 0 else "",
                lines=piece.split("\n"),
                start=start,
                end=start + len(piece),
            )
        )
        cursor = (found if found >= 0 else cursor) + len(piece)
    return blocks


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in RE_SENTENCE.split(paragraph):
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces
