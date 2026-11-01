"""Формулы из PDF: детекция, кроп в PNG, хранение в SQLite, описание.

Формулы в PDF нормативов набраны двумя способами, и оба требуют отдельной
детекции — pymupdf4llm их в Markdown не превращает:

1. **Математический шрифт** (SymbolMT): скобки, знаки, Σ живут в PUA-кодах
   ``\\ue000-\\uf8ff`` — строка формулы содержит несколько таких глифов подряд.
   Символы единиц (°, ², ³) тоже из этого диапазона, но в формулах их роль
   ничтожна — при завязке на них строка «Вт/(м² °С)» в тексте была бы ложной
   формулой, поэтому они исключены из подсчёта.
2. **Отдельный номер-тег**: строка целиком вида ``(А.6)`` справа от формулы.
   Регион формулы — та же строка слева от тега (с расширением вверх, пока
   сверху есть чертёж/математические глифы). Тег без единого числа,
   оператора, глифа или чертежа — почти наверняка ссылка, а не формула.

Каждая формула сохраняется PNG-кропом в ``data/img/<код>/`` (там же лежат
картинки figures) и строкой в таблице ``formulas``: страница, номер, якорный
текст («…по формуле:»), а после описания — LaTeX/вычисляемое выражение/
параметры/проза. Описание, как и у картинок, вшивается меткой в чанк
(``<!-- fml:N -->``), чтобы находиться FTS5 и вектором рядом со своим пунктом.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from . import config as _config  # IMG_DIR читаем динамически: тесты его патчат
from .mathlayout import latex_from_region

#: PUA-коды (MathType/SymbolMT) и математические операторы без единиц.
PUA_RE = re.compile(r"[\ue000-\uf8ff]")
#: Символы единиц/степеней в PUA: °C, м², м³ — это текст, а не формула.
PUA_UNITS = {"\uf0b0", "\uf0b1", "\uf0b2", "\uf0b3"}

#: Отдельная строка-номер формулы: «(А.6)», «(1.2)» — без ничего вокруг.
TAG_RE = re.compile(r"^\(([А-П]\.\d{1,2}|\d{1,2}(?:\.\d{1,2}){1,2})\)$")

#: Математический признак региона: цифра или оператор (в тексте PDF их нет,
#: в отличие от Markdown-звёздочек).
MATH_HINT = re.compile(r"[=+×÷*/·]|\d")

#: Максимальная высота региона формулы над тегом.
MAX_BAND_ABOVE = 45.0
#: Сколько раз регион может вырасти вверх (двухстрочные формулы).
MAX_EXPAND_UP = 3


@dataclass(slots=True)
class FormulaCandidate:
    """Формула, найденная в PDF, ещё не записанная в базу."""

    page: int                      #: номер страницы, с 1
    tag: str | None                #: номер формулы без скобок («А.6») или None
    bbox: tuple[float, float, float, float]
    anchor: str                    #: ближайший текст сверху («…по формуле:»)
    path: Path                     # куда сохранён кроп
    latex_geom: str = ""           #: LaTeX, восстановленный по геометрии символов


def extract_formulas(
    pdf_path: Path | str, doc_code: str, img_dir: Path | str | None = None
) -> list[FormulaCandidate]:
    """Вырезает формулы из PDF в ``img_dir`` (по умолчанию ``data/img/<slug>``)."""
    import pymupdf

    img_dir = Path(img_dir) if img_dir else _config.IMG_DIR / _slug(doc_code)
    img_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path))
    found: list[FormulaCandidate] = []
    try:
        for pno in range(len(doc)):
            page = doc[pno]
            lines = _page_lines(page)
            if not lines:
                continue
            drawings = [pymupdf.Rect(item["rect"]) for item in page.get_drawings()]
            tag_regions = _tag_regions(page, lines, drawings)
            tagged, clusters = _assign_clusters(tag_regions, _pua_clusters(lines))
            regions = tagged + _merge_regions(clusters)
            regions = [r for r in regions if _has_math_signal(page, lines, drawings, r)]
            regions.sort(key=lambda r: (r[1], r[0]))
            saved = 0
            for region in regions:
                clip = _clamp(region, page.rect)
                if clip is None:
                    continue
                saved += 1
                path = img_dir / f"f_p{pno + 1:03d}_{saved:02d}.png"
                pix = page.get_pixmap(clip=clip, dpi=200)
                if pix.colorspace and pix.colorspace.n > 3:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                pix.save(str(path))
                found.append(
                    FormulaCandidate(
                        page=pno + 1,
                        tag=region[4],
                        bbox=(round(clip.x0, 1), round(clip.y0, 1), round(clip.x1, 1), round(clip.y1, 1)),
                        anchor=_anchor(lines, clip, doc_code),
                        path=path,
                        latex_geom=latex_from_region(page, clip, region[4]),
                    )
                )
    finally:
        doc.close()
    return found


def sync_formulas(
    store: Any,
    doc_id: int,
    pdf_path: Path | str,
    doc_code: str,
    *,
    verbose: bool = True,
) -> int:
    """Пересыпает формулы PDF в таблицу ``formulas`` и возвращает их число.

    Идемпотентно: имя кропа детерминировано (страница + порядок на странице),
    повторный вызов не плодит дубли и не трогает записанные описания.
    Строки, чьи файлы больше не производятся (изменился PDF), удаляются.
    Описания прошлой индексации заново вшиваются в чанки
    (``reapply_formula_notes`` — replace_chunks их вырезал).
    """
    candidates = extract_formulas(pdf_path, doc_code)
    paths: list[str] = []
    for item in candidates:
        store.upsert_formula(
            doc_id=doc_id,
            tag=item.tag,
            page=item.page,
            bbox=",".join(str(value) for value in item.bbox),
            path=str(item.path),
            anchor=item.anchor,
            latex_geom=item.latex_geom,
        )
        paths.append(str(item.path))
    store.drop_missing_formulas(doc_id, keep=paths)
    described = store.reapply_formula_notes(doc_id)
    store.commit()
    if verbose and candidates:
        print(
            f"[fml] {len(candidates)} формул, {described} с описанием -> "
            f"{_config.IMG_DIR / _slug(doc_code)}",
            flush=True,
        )
    return len(candidates)


# ---------------------------------------------------------------- markdown
#: Маркеры блока формулы в Markdown: по ним блок снимается и ставится заново,
#: поэтому повторная вставка идемпотентна (текст не дублируется).
MD_OPEN = "<!-- fml-md:{key} -->"
MD_CLOSE = "<!-- /fml-md:{key} -->"
MD_EXTRA_OPEN = "<!-- fml-md-extra -->"
MD_EXTRA_CLOSE = "<!-- /fml-md-extra -->"

#: Прежние блоки (в том числе от прошлой индексации) — DOTALL, без жадности.
_MD_BLOCK_RE = re.compile(
    r"\n?<!-- fml-md:[^>\n]*-->\n.*?\n<!-- /fml-md:[^>\n]*-->\n?", re.S
)
_MD_EXTRA_RE = re.compile(
    r"\n?<!-- fml-md-(?:extra|untagged) -->\n.*?"
    r"\n<!-- /fml-md-(?:extra|untagged) -->\n?",
    re.S,
)
_HTML_TAG = re.compile(r"<[^>]+>")


def formula_markdown_block(row: Any, md_dir: Path | str | None = None) -> str:
    """Тело блока формулы: LaTeX > вычисляемое выражение > кроп PNG + описание.

    В Markdown норматив идёт вместо сплющенной строки PDF. Пока формула не
    распознана, ставится ссылка на кроп — это честнее, чем потерять её.
    """
    latex = _field(row, "latex")
    geom = "" if latex else _sane_geometry(_field(row, "latex_geom"))
    expr = _field(row, "expr")
    description = _field(row, "description")
    tag = _field(row, "tag")
    label = f"({tag})" if tag else "без номера"
    lines: list[str] = []
    if latex:
        # распознано моделью/агентом — самый надёжный вариант
        lines += ["$$", latex, "$$"]
    elif geom:
        # восстановлено по геометрии символов: структура верна, опечатки
        # возможны, поэтому рядом остаётся кроп для сверки глазами
        lines += ["$$", geom, "$$"]
        href = _image_href(row, md_dir)
        if href:
            lines.append(f"![Формула {label} — сверка с PDF]({href})")
    elif expr:
        lines += ["```text", expr, "```"]
    else:
        href = _image_href(row, md_dir)
        if not href:
            return ""
        lines.append(f"![Формула {label}]({href})")
    if description:
        lines.append(f"*{description}*")
    return "\n".join(lines)


def _sane_geometry(latex: str) -> str:
    """Геометрический LaTeX без грубых дефектов; "" — показывать его нельзя.

    Черновое восстановление не должно подменять картинку мусором. Признаки
    брака: обрывки объяснений внутри ``\\text{…}`` (в СП это куски абзаца
    «-ного помещения, определяемый…»), несбалансированные скобки, отсутствие
    отношения или дроби, неразобранные глифы.
    """
    if len(latex) < 4 or len(latex) > 300:
        return ""
    if latex.count("{") != latex.count("}"):
        return ""
    if any(0xE000 <= ord(char) <= 0xF8FF for char in latex):
        return ""
    if not re.search(r"[=<>]|\\frac|\\le|\\ge|\\approx", latex):
        return ""
    if latex.startswith(("^", "_")):
        return ""              # индекс без основания — строка разобрана не с начала
    if re.search(r"[=<>]\s*[\^_]", latex):
        return ""              # правая часть отношения потеряна
    if re.search(r"(?:^|[^\d]),\d", latex):
        return ""              # «,075» — ведущая цифра потеряна
    if re.search(r"[ÂâÎîÁáÃÑÏòî]", latex):
        return ""              # битая кодировка шрифта (кириллица как Latin-1)
    for chunk in re.findall(r"\\text\{([^{}]*)\}", latex):
        if len(chunk) > 6:      # индекс короткий, фраза — нет
            return ""
    return latex


def patch_markdown(
    text: str, formulas: Sequence[Any], md_dir: Path | str | None = None
) -> tuple[str, int]:
    """Вставляет блоки формул в Markdown, не теряя ни одной формулы.

    Порядок привязки — от точной к грубой:

    1. строка-номер отдельной строкой (``(А.6)``) — блок ставится над ней;
       строка остаётся на месте: по ней формулу находит чанк-носитель
       (``store._formula_chunk``);
    2. якорный текст над формулой («…определяется по формуле») — блок после
       этой строки: так формула встаёт туда, где PDF потерял её тело;
    3. номер внутри строки («…, (Б.1)») — блок после неё;
    4. ни одного признака — раздел в конце документа, чтобы не потерять.

    Идемпотентно: прежние блоки и раздел снимаются по маркерам и ставятся
    заново, повторный вызов даёт тот же текст. Возвращает
    ``(новый текст, сколько блоков вставлено)``.
    """
    text = _MD_EXTRA_RE.sub("\n", text)
    text = _MD_BLOCK_RE.sub("\n", text)
    # снятие блока оставляет лишнюю пустую строку — схлопываем её, иначе
    # повторная вставка каждый раз добавляла бы ещё одну
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = text.split("\n")

    pending: list[dict[str, str]] = []
    for row in formulas:
        block = _md_block(row, md_dir)
        if block:
            pending.append({
                "tag": _field(row, "tag"),
                "block": block,
                "anchor": _field(row, "anchor"),
            })

    before: dict[int, list[str]] = {}
    after: dict[int, list[str]] = {}
    extra: list[str] = []

    # 1) номер формулы — отдельная строка
    for item in list(pending):
        needle = f"({item['tag']})" if item["tag"] else ""
        if not needle:
            continue
        for index, line in enumerate(lines):
            if line.strip() == needle:
                before.setdefault(index, []).append(item["block"])
                pending.remove(item)
                break

    # 2) якорный текст над формулой
    for item in list(pending):
        index = _anchor_index(lines, item["anchor"])
        if index is not None:
            after.setdefault(index, []).append(item["block"])
            pending.remove(item)

    # 3) номер внутри строки — сплющенная формула осталась в ней
    for item in list(pending):
        needle = f"({item['tag']})" if item["tag"] else ""
        if not needle:
            continue
        for index, line in enumerate(lines):
            if line.lstrip().startswith("|"):
                continue
            if needle in line:
                after.setdefault(index, []).append(item["block"])
                pending.remove(item)
                break

    # 4) привязать не к чему — раздел в конце документа
    for item in pending:
        extra.append(item["block"])
        if item["tag"]:
            extra.extend([f"({item['tag']})", ""])

    inserted = sum(len(blocks) for blocks in before.values())
    inserted += sum(len(blocks) for blocks in after.values())
    inserted += sum(1 for item in pending)

    out: list[str] = []
    for index, line in enumerate(lines):
        for block in before.get(index, []):
            out.extend([block, ""])
        out.append(line)
        for block in after.get(index, []):
            out.extend(["", block])
    text = "\n".join(out)

    if extra:
        appendix = [MD_EXTRA_OPEN, "## Формулы, извлечённые из PDF", ""]
        appendix += extra
        appendix.append(MD_EXTRA_CLOSE)
        text = text.rstrip() + "\n\n" + "\n".join(appendix) + "\n"
    return text, inserted


def patch_document_md(
    store: Any,
    doc_code: str,
    md_dir: Path | str | None = None,
    *,
    verbose: bool = True,
) -> int | None:
    """Переписывает сохранённый Markdown документа блоками формул.

    ``None`` — файла нет (артефакт удалён), ``int`` — сколько блоков вставлено.
    """
    doc = store.get_document(doc_code)
    if doc is None or not doc["md_path"]:
        return None
    md_path = Path(doc["md_path"])
    if not md_path.exists():
        return None
    original = md_path.read_text(encoding="utf-8")
    patched, inserted = patch_markdown(
        original, store.list_formulas(doc_code), md_dir or md_path.parent
    )
    if patched != original:
        md_path.write_text(patched, encoding="utf-8")
    if verbose:
        print(f"[fml] {inserted} блоков формул -> {md_path.name}", flush=True)
    return inserted


def _md_block(row: Any, md_dir: Path | str | None) -> str:
    body = formula_markdown_block(row, md_dir)
    if not body:
        return ""
    key = _field(row, "tag") or f"id{_field(row, 'formula_id')}"
    return f"{MD_OPEN.format(key=key)}\n{body}\n{MD_CLOSE.format(key=key)}"


def _field(row: Any, name: str) -> str:
    """Значение поля строки sqlite3.Row/dict строкой (нет поля — пусто)."""
    try:
        value = row[name]
    except (IndexError, KeyError):
        return ""
    return "" if value is None else str(value).strip()


def _image_href(row: Any, md_dir: Path | str | None) -> str:
    """Путь кропа относительно каталога Markdown (posix-разделители)."""
    raw = _field(row, "path")
    if not raw:
        return ""
    path = Path(raw)
    if md_dir:
        try:
            return Path(os.path.relpath(path, Path(md_dir))).as_posix()
        except ValueError:  # разные диски Windows — оставляем абсолютный путь
            pass
    return path.as_posix()


def _match_key(value: str) -> str:
    """Текст для сравнения якоря с Markdown: без тегов, разметки и лишних пробелов."""
    text = _HTML_TAG.sub(" ", value or "")
    text = text.replace("*", " ").replace("_", " ").replace("\\", " ")
    return re.sub(r"\s+", " ", text).strip()


def _anchor_index(lines: Sequence[str], anchor: str) -> int | None:
    """Строка Markdown, в которой узнаётся якорный текст над формулой.

    Якорь склеен из строк PDF («ближайшая» + строка над ней), а Markdown
    переносит текст иначе, поэтому пробуем несколько окон: сначала начало
    (ближайшая строка), затем конец («…по формуле»).
    """
    needle = _match_key(anchor)
    if len(needle) < 25:
        return None
    keys = [_match_key(line) for line in lines]

    def hits(probe: str) -> list[int]:
        # строки таблиц — это оглавление и перечни: якорь там встречается
        # раньше самого раздела, и формула уехала бы в содержание
        return [
            index
            for index, key in enumerate(keys)
            if not lines[index].lstrip().startswith("|") and probe in key
        ]

    # начало якоря — это строка прямо над формулой: сначала самое длинное окно
    for size in (120, 90, 60):
        if len(needle) <= size:
            continue
        found = hits(needle[:size])
        if found:
            return found[0]
    # конец якоря («…определяется по формуле») — фраза нередко встречается
    # в разных местах, поэтому берём только однозначное совпадение
    tail = needle[-40:] if len(needle) > 40 else needle
    found = hits(tail) if len(tail) >= 25 else []
    if len(found) == 1:
        return found[0]
    return None


# ------------------------------------------------------------------ vision
#: JSON-схема, которую обязана вернуть vision-модель (см. describe_formula_with_llm).
FORMULA_PROMPT = (
    "Ты распознаёшь формулу из нормативного документа по строительству "
    "(СП/ГОСТ) по вырезанному изображению. Верни СТРОГО один JSON-объект "
    "без пояснений и без markdown: "
    '{"latex": "формула в LaTeX", "expr": "вычисляемое выражение на языке '
    'Python (имена переменных латиницей, кириллические индексы раскладываются '
    'вроде t_h)", "params": {"символ": "расшифровка с единицами измерения"}, '
    '"text": "1-3 предложения: что считает формула"}. '
    "Пустые поля — пустая строка, params при отсутствии — пустой объект. "
    "Если формулу не разобрать — верни объект с пустыми latex/expr и text "
    "«неразборчиво»."
)


def describe_formula_with_llm(
    image_path: Path | str, anchor: str, tag: str | None, cfg: Any
) -> dict[str, Any]:
    """Кроп формулы -> структурированное описание через chat/completions.

    Возвращает ``{"latex", "expr", "params", "text"}``; ``ValueError``, если
    модель не вернула валидный JSON (reasoning-модели съедают бюджет на
    размышления — запрос повторяется с двойным ``max_tokens``, как в figures).
    """
    import httpx

    from .figures import MIN_DESC_CHARS, _png_bytes, _parse_description

    if not cfg.auto:
        raise ValueError("vision.model/base_url/api_key не заданы — автоописание выключено")
    png = _png_bytes(Path(image_path), cfg.max_dim)
    import base64

    b64 = base64.b64encode(png).decode("ascii")
    context = f"Номер формулы: ({tag})" if tag else "Номер формулы не указан"
    if anchor:
        context += f"\nТекст документа рядом: {anchor}"
    last_error = "нет попыток"
    for max_tokens in (cfg.max_tokens, cfg.max_tokens * 2):
        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": FORMULA_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                },
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
        }
        try:
            response = httpx.post(
                f"{cfg.base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {cfg.api_key}"},
                timeout=cfg.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        raw = _parse_description(response.json(), cfg.model)
        data = _parse_formula_json(raw)
        if data is None:
            last_error = f"не JSON: {raw[:120]!r}"
            continue
        if len(data.get("text", "")) >= MIN_DESC_CHARS or data.get("latex"):
            return data
        last_error = "пустое описание (неразборчиво или бюджет размышлений)"
    raise ValueError(f"LLM не вернула описание формулы (модель {cfg.model}): {last_error}")


def _parse_formula_json(raw: str) -> dict[str, Any] | None:
    """Достаёт объект из ответа модели: снимает ```json-ограждение, ищет {...}."""
    if not raw:
        return None
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    params = data.get("params")
    if not isinstance(params, dict):
        params = {}
    return {
        "latex": str(data.get("latex") or "").strip(),
        "expr": str(data.get("expr") or "").strip(),
        "params": {str(k): str(v) for k, v in params.items()},
        "text": str(data.get("text") or "").strip(),
    }


# ------------------------------------------------------------------ internal
def _page_lines(page: Any) -> list[dict[str, Any]]:
    """Текстовые строки страницы: текст, bbox, счётчик математических PUA."""
    lines: list[dict[str, Any]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            pua = sum(
                1
                for span in spans
                for char in span.get("text", "")
                if PUA_RE.match(char) and char not in PUA_UNITS
            )
            lines.append({"text": text, "bbox": tuple(line["bbox"]), "pua": pua})
    return lines


def _tag_regions(page: Any, lines: list[dict], drawings: list) -> list[tuple]:
    """Регионы формул по отдельным строкам-тегам: «формула слева от (А.6)»."""
    import pymupdf

    regions: list[tuple] = []
    for line in lines:
        match = TAG_RE.match(line["text"])
        if not match:
            continue
        tag_box = pymupdf.Rect(line["bbox"])
        y0 = tag_box.y0 - MAX_BAND_ABOVE
        y1 = tag_box.y1 + 10
        x1 = tag_box.x1 + 6
        # левая граница — самая левая строка в полосе тега (формула тянется
        # от поля до номера); без строк — поле страницы
        in_band = [
            item
            for item in lines
            if pymupdf.Rect(item["bbox"]).y1 >= y0 and pymupdf.Rect(item["bbox"]).y0 <= y1
        ]
        x0 = min((pymupdf.Rect(item["bbox"]).x0 for item in in_band), default=page.rect.x0 + 20)
        band = pymupdf.Rect(max(x0 - 4, page.rect.x0 + 10), y0, min(x1, page.rect.x1 - 15), y1)
        # вытягиваем вверх, пока сверху есть чертёж или мат. глифы —
        # так двухстрочные формулы не режутся
        for _ in range(MAX_EXPAND_UP):
            grow = _above_content(band, lines, drawings)
            if grow is None:
                break
            band.y0 = min(band.y0, grow)
        # кластер-формула достаётся одному тегу в _assign_clusters; здесь
        # регион только стягивается к границам страницы
        band.y0 = max(band.y0, page.rect.y0 + 5)
        regions.append((band.x0, band.y0, band.x1, band.y1, match.group(1)))
    return regions


def _assign_clusters(
    tag_regions: list[tuple], clusters: list[tuple]
) -> tuple[list[tuple], list[tuple]]:
    """Распределяет PUA-кластеры: каждый — ровно одному теговому региону.

    Теговые регионы НЕ сливаются друг с другом: каждый тег — отдельная
    формула, а полосы соседних тегов пересекаются (спейсинг номеров меньше
    высоты полосы) — при слиянии часть номеров терялась. Кластер достаётся
    тегу с наибольшей площадью пересечения; без пересечения — ближайшему
    по вертикали (формула, вылезшая за полосу). Совсем чужой кластер
    идёт отдельной формулой без номера.
    """
    import pymupdf

    assigned = list(tag_regions)
    leftovers: list[tuple] = []
    for cluster in clusters:
        box = pymupdf.Rect(cluster)
        box.x0 -= 8
        box.y0 -= 6
        box.x1 += 8
        box.y1 += 6
        best_index = -1
        best_area = 0.0
        for index, region in enumerate(assigned):
            region_box = pymupdf.Rect(region[0], region[1], region[2], region[3])
            if not box.intersects(region_box):
                continue
            area = (box & region_box).get_area()
            if area > best_area:
                best_area = area
                best_index = index
        if best_index < 0:
            nearest: int | None = None
            nearest_gap: float | None = None
            for index, region in enumerate(assigned):
                region_box = pymupdf.Rect(region[0], region[1], region[2], region[3])
                if box.x1 < region_box.x0 - 10 or region_box.x1 < box.x0 - 10:
                    continue
                gap = max(region_box.y0 - box.y1, box.y0 - region_box.y1, 0.0)
                if nearest_gap is None or gap < nearest_gap:
                    nearest_gap = gap
                    nearest = index
            if nearest is None or (nearest_gap is not None and nearest_gap > 30):
                leftovers.append((box.x0, box.y0, box.x1, box.y1, None))
                continue
            best_index = nearest
        region = assigned[best_index]
        assigned[best_index] = (
            min(region[0], box.x0),
            min(region[1], box.y0),
            max(region[2], box.x1),
            max(region[3], box.y1),
            region[4],
        )
    return assigned, leftovers


def _merge_regions(regions: list[tuple]) -> list[tuple]:
    """Слияние пересекающихся регионов (кластер внутри тегового и т.п.)."""
    merged: list[tuple] = []
    changed = False
    for region in regions:
        box = (region[0], region[1], region[2], region[3])
        hit = None
        for index, other in enumerate(merged):
            if _overlap(box, (other[0], other[1], other[2], other[3])):
                hit = index
                break
        if hit is None:
            merged.append(region)
            continue
        base = merged[hit]
        union = (
            min(base[0], region[0]),
            min(base[1], region[1]),
            max(base[2], region[2]),
            max(base[3], region[3]),
            base[4] or region[4],
        )
        merged[hit] = union
        changed = True
    # вторая проходка: после слияния регионы могли стать соседними
    if changed and len(merged) > 1:
        return _merge_regions(merged)
    return merged


def _pua_clusters(lines: list[dict]) -> list[tuple[float, float, float, float]]:
    """Строки с мат. глифами, объединённые в кластеры (многострочные формулы)."""
    import pymupdf

    boxes = [pymupdf.Rect(line["bbox"]) for line in lines if line["pua"] >= 2]
    clusters: list[Any] = []
    for box in boxes:
        target = None
        for cluster in clusters:
            grown = pymupdf.Rect(cluster)
            grown.x0 -= 8
            grown.y0 -= 8
            grown.x1 += 8
            grown.y1 += 8
            if box.intersects(grown):
                target = cluster
                break
        if target is None:
            clusters.append(pymupdf.Rect(box))
        else:
            target |= box
    return [(c.x0, c.y0, c.x1, c.y1) for c in clusters]


def _above_content(band: Any, lines: list[dict], drawings: list) -> float | None:
    """Ниже какого y заканчивается «формульный» материал сразу над регионом."""
    best: float | None = None
    for rect in drawings:
        if rect.y1 <= band.y0 and rect.y1 >= band.y0 - 40 and rect.x1 >= band.x0 and rect.x0 <= band.x1:
            # черта дроби (короткая), а не линия сетки таблицы/колонки
            if 12 < rect.x1 - rect.x0 < 160:
                best = rect.y0 if best is None else min(best, rect.y0)
    for line in lines:
        if line["pua"] >= 1:
            box = line["bbox"]
            if box[3] <= band.y0 and box[3] >= band.y0 - 40 and box[2] >= band.x0 and box[0] <= band.x1:
                best = box[1] if best is None else min(best, box[1])
    return best


def _has_math_signal(page: Any, lines: list[dict], drawings: list, region: tuple) -> bool:
    """False — регион без единого признака формулы (ссылка-тег, а не уравнение)."""
    import pymupdf

    box = pymupdf.Rect(region[0], region[1], region[2], region[3])
    pua = 0
    text_parts = []
    for line in lines:
        line_box = pymupdf.Rect(line["bbox"])
        if line_box.intersects(box):
            pua += line["pua"]
            # сам тег «(А.6)» не источник сигнала: его цифры делали бы
            # проверку самовыполняющейся для любой случайной строки-ссылки
            if not TAG_RE.match(line["text"]):
                text_parts.append(line["text"])
    if pua:
        return True
    if any(rect.intersects(box) for rect in drawings):
        return True
    return bool(MATH_HINT.search(" ".join(text_parts)))


def _anchor_ok(text: str, doc_code: str) -> bool:
    """Строка пригодна как контекст: не колонтитул и не отметка редакции."""
    if doc_code and doc_code in text:
        return False
    lowered = text.lower()
    return not ("проект" in lowered and "редакция" in lowered)


def _anchor(lines: list[dict], clip: Any, doc_code: str = "") -> str:
    """Ближайшие текстовые строки над формулой («…определяется по формуле:»)."""
    above = [
        line
        for line in lines
        if _anchor_ok(line["text"], doc_code)
        and len(line["text"]) >= 12  # фрагменты формул («с.ов Q») — не контекст
        and line["bbox"][3] <= clip.y0 + 2
        and line["bbox"][3] >= clip.y0 - 90
        and line["bbox"][2] >= clip.x0 - 40
        and line["bbox"][0] <= clip.x1 + 40
    ]
    if not above:
        return ""
    above.sort(key=lambda line: line["bbox"][3], reverse=True)
    return " ".join(line["text"] for line in above[:2])[:400]


def _clamp(rect: tuple, page_rect: Any) -> Any | None:
    """Прямоугольник кропа в границах страницы; None — слишком мал."""
    import pymupdf

    clip = pymupdf.Rect(
        max(rect[0], 0.0),
        max(rect[1], 0.0),
        min(rect[2], page_rect.x1),
        min(rect[3], page_rect.y1),
    )
    if clip.width < 40 or clip.height < 15:
        return None
    return clip


def _overlap(a: tuple, b: tuple) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _slug(code: str) -> str:
    return re.sub(r"[^\w.]+", "_", code, flags=re.UNICODE).strip("_") or "document"
