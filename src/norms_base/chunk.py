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
RE_CLAUSE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+\S")
RE_TABLE = re.compile(r"^\|")
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

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip("\n")


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
        if current is not None and current.lines:
            current.end = offset
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
            section, section_title = _section_of(heading)
            pending_heading = heading
            continue

        if RE_TABLE.match(stripped):
            if current is not None and current.kind != "table":
                flush()
            if current is None:
                current = _new_block("table", line_start, section, section_title, pending_heading)
                pending_heading = ""
            current.lines.append(raw_line)
            continue

        clause_match = RE_CLAUSE.match(stripped)
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
    return _oversized(merged, max_chars)


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
        if RE_CLAUSE.match(inner) is None and "\n" not in inner:
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
    """Дотягивает мелкие блоки до min_chars: вперёд в пределах раздела, затем назад."""
    forward: list[_Block] = []
    for block in blocks:
        if (
            forward
            and block.size < min_chars
            and not block.heading
            and forward[-1].section == block.section
            and forward[-1].size + block.size <= min_chars * 2
        ):
            target = forward[-1]
            target.lines.extend([""] + block.lines)
            target.end = block.end
            target.clause_id = target.clause_id or block.clause_id
            if target.kind != block.kind:
                target.kind = "text" if block.kind == "text" else target.kind
            continue
        forward.append(block)

    backward: list[_Block] = []
    for block in forward:
        if (
            backward
            and block.size < min_chars
            and backward[-1].section == block.section
            and not block.heading
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
    """Разрезает блоки больше max_chars по абзацам/предложениям (смещения сохраняются)."""
    result: list[_Block] = []
    for block in blocks:
        if block.size <= max_chars:
            result.append(block)
            continue
        result.extend(_split_block(block, max_chars))
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
