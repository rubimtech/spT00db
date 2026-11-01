"""Восстановление LaTeX формулы по геометрии символов PDF.

Текстовый слой PDF теряет структуру формулы: числитель и знаменатель идут
подряд, степени и индексы неотличимы от строки, глифы SymbolMT превращаются
в PUA-мусор. Но у каждого фрагмента есть координаты, размер и шрифт — этого
достаточно, чтобы собрать обратно дроби (по черте дроби в ``get_drawings``),
верхние/нижние индексы (по кеглю и смещению базовой линии) и скобки
(по глифам-расширениям SymbolMT).

Модуль не «понимает» математику: он восстанавливает то, что видно на
странице. Сложные многострочные выкладки (матрицы, системы) получаются
приблизительно — это всё равно лучше сплющенной строки, а точное
распознавание остаётся за vision-моделью (``formulas --auto``).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable

#: Глифы SymbolMT -> LaTeX. Ключ — код символа в шрифте Symbol (0x20..0xFF).
SYMBOL_MAP: dict[int, str] = {
    0x22: r"\forall", 0x24: r"\exists", 0x27: r"\ni", 0x2A: "*",
    0x2B: "+", 0x2D: "-", 0x3D: "=", 0x40: r"\cong", 0x5E: r"\perp",
    0x7E: r"\sim", 0xA1: r"\Upsilon", 0xA2: r"\prime", 0xA3: r"\le",
    0xA4: "/", 0xA5: r"\infty", 0xA6: r"\varphi", 0xAB: r"\leftrightarrow",
    0xAC: r"\leftarrow", 0xAD: r"\uparrow", 0xAE: r"\rightarrow",
    0xAF: r"\downarrow", 0xB0: r"^{\circ}", 0xB1: r"\pm", 0xB2: r"\prime\prime",
    0xB3: r"\ge", 0xB4: r"\times", 0xB5: r"\propto", 0xB6: r"\partial",
    0xB7: r"\cdot", 0xB8: r"\div", 0xB9: r"\ne", 0xBA: r"\equiv",
    0xBB: r"\approx", 0xBC: r"\ldots", 0xC4: r"\otimes", 0xC5: r"\oplus",
    0xC6: r"\varnothing", 0xC7: r"\cap", 0xC8: r"\cup", 0xC9: r"\supset",
    0xCA: r"\supseteq", 0xCB: r"\not\subset", 0xCC: r"\subset",
    0xCD: r"\subseteq", 0xCE: r"\in", 0xCF: r"\notin", 0xD0: r"\angle",
    0xD1: r"\nabla", 0xD5: r"\prod", 0xD6: r"\surd", 0xD7: r"\cdot",
    0xD8: r"\neg", 0xD9: r"\wedge", 0xDA: r"\vee", 0xDB: r"\Leftrightarrow",
    0xDC: r"\Leftarrow", 0xDD: r"\Uparrow", 0xDE: r"\Rightarrow",
    0xDF: r"\Downarrow", 0xE0: r"\diamond", 0xE1: "<", 0xE5: r"\sum",
    0xF1: ">", 0xF2: r"\int", 0xBD: "|",
}

#: Греческие строчные SymbolMT (0x61..0x7A) -> команды LaTeX.
GREEK = [
    "alpha", "beta", "chi", "delta", "epsilon", "phi", "gamma", "eta",
    "iota", "phi1", "kappa", "lambda", "mu", "nu", "omicron", "pi",
    "theta", "rho", "sigma", "tau", "upsilon", "omega1", "omega", "xi",
    "psi", "zeta",
]
for _index, _name in enumerate(GREEK):
    SYMBOL_MAP.setdefault(0x61 + _index, "\\" + _name)

#: Глифы-расширения скобок: (код, символ, открывающая ли).
BRACKET_MAP: dict[int, tuple[str, bool]] = {
    0xE6: ("(", True), 0xE7: ("(", True), 0xE8: ("(", True),
    0xE9: (")", False), 0xEA: (")", False), 0xEB: (")", False),
    0xF6: ("[", True), 0xF7: ("[", True), 0xF8: ("[", True),
    0xF9: ("{", True), 0xFA: ("{", True), 0xFB: ("{", True),
    0xEC: ("}", False), 0xED: ("}", False), 0xEE: ("}", False),
    0xF3: ("]", False), 0xF4: ("]", False), 0xF5: ("]", False),
}

#: Функции, которые в LaTeX пишутся командой, а не курсивными буквами.
FUNCTIONS = {
    "cos": r"\cos", "sin": r"\sin", "tg": r"\tg", "ctg": r"\ctg",
    "tan": r"\tan", "cot": r"\cot", "exp": r"\exp", "ln": r"\ln",
    "lg": r"\lg", "log": r"\log", "min": r"\min", "max": r"\max",
    "arcsin": r"\arcsin", "arccos": r"\arccos", "arctg": r"\arctg",
}

CYRILLIC = re.compile(r"[А-Яа-яЁё]")
MATH_TAG = re.compile(r"^\(\s*[А-ПA-Z]?\.?\d{1,2}(?:\.\d{1,2}){0,2}\s*\)$")
#: Строка прозы (объяснение вокруг формулы), а не фрагмент выкладки.
PROSE = re.compile(r"[А-Яа-яЁё]{4,}[ ,][^=]{10,}")


@dataclass
class Atom:
    """Один фрагмент выкладки: символ, число, оператор, скобка или дробь."""

    latex: str
    x0: float
    x1: float
    y0: float
    y1: float
    base: float          #: базовая линия (origin y)
    size: float          #: кегль
    kind: str = "text"   #: text | op | rel | open | close | frac
    math: bool = True    #: похоже на выкладку, а не на объяснение вокруг

    @property
    def width(self) -> float:
        return self.x1 - self.x0


@dataclass
class Row:
    """Строка выкладки: атомы с одной базовой линией."""

    atoms: list[Atom] = field(default_factory=list)

    @property
    def base(self) -> float:
        return sum(atom.base for atom in self.atoms) / len(self.atoms)

    @property
    def size(self) -> float:
        return max(atom.size for atom in self.atoms)


# --------------------------------------------------------------------- вход
def latex_from_region(page: Any, clip: Any, tag: str | None = None) -> str:
    """LaTeX формулы в прямоугольнике страницы; "" — разобрать не удалось."""
    import pymupdf

    atoms, bars = _collect(page, pymupdf.Rect(clip), tag or "")
    if not atoms:
        return ""
    latex = _compose(atoms, bars, _base_size(atoms))
    return _tidy(latex)


def _collect(page: Any, clip: Any, tag: str) -> tuple[list[Atom], list[tuple[float, float, float]]]:
    """Атомы и черты дробей внутри региона (без прозы и номера формулы).

    Регион кропа намеренно широкий: в него попадают объяснение над формулой и
    соседние строки. Поэтому атомы сначала складываются в строки по базовой
    линии, а затем берётся непрерывная полоса строк-выкладок вокруг строки с
    номером формулы: строка-объяснение эту полосу обрывает.
    """
    atoms: list[Atom] = []
    brackets: list[Atom] = []
    tag_base: float | None = None
    data = page.get_text("rawdict", clip=clip)
    for block in data["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = "".join(char["c"] for char in span["chars"])
                if not text.strip():
                    continue
                if tag and _is_tag(text, tag):
                    tag_base = span["origin"][1]
                    continue
                plain, found = _span_atoms(span, text)
                atoms.extend(plain)
                brackets.extend(found)
    atoms.extend(_one_bracket_per_x(brackets))

    rows = _rows(atoms)
    rows = _formula_rows(rows, tag_base)
    atoms = _dedupe([atom for row in rows for atom in row.atoms if atom.math])
    if not atoms:
        return [], []

    top = min(row.base for row in rows) - 12
    bottom = max(row.base for row in rows) + 6

    bars: list[tuple[float, float, float]] = []
    for drawing in page.get_drawings():
        rect = drawing["rect"]
        if rect.height > 2.5 or rect.width < 8:
            continue  # не черта дроби (рамка, линия таблицы)
        y = (rect.y0 + rect.y1) / 2
        if not (top <= y <= bottom):
            continue
        if rect.x1 < clip.x0 or rect.x0 > clip.x1:
            continue
        bars.append((rect.x0, rect.x1, y))
    return atoms, bars


def _dedupe(atoms: list[Atom]) -> list[Atom]:
    """Убирает наложения: один знак часто нарисован двумя шрифтами.

    «=», «·», «−» в СП набраны и обычным шрифтом, и SymbolMT — в текстовом
    слое это два фрагмента почти на одном месте. Оставляем первый по порядку
    чтения, остальные наложения отбрасываем.
    """
    result: list[Atom] = []
    for atom in sorted(atoms, key=lambda item: (item.base, item.x0)):
        duplicate = False
        for kept in result:
            if abs(kept.base - atom.base) > 1.5:
                continue
            overlap = min(kept.x1, atom.x1) - max(kept.x0, atom.x0)
            if overlap > 0.6 * min(kept.width, atom.width):
                duplicate = True
                break
        if not duplicate:
            result.append(atom)
    return result


def _rows(atoms: list[Atom]) -> list[Row]:
    """Группирует атомы по базовой линии; мелкие кегли — к ближайшей строке.

    Индекс стоит на своей базовой линии, на 3–5 pt ниже/выше основной, поэтому
    строку определяют крупные атомы, а мелкие приписываются к ближайшей.
    """
    rows: list[Row] = []
    for atom in sorted(atoms, key=lambda item: (-item.size, item.x0)):
        best: Row | None = None
        best_distance: float | None = None
        for row in rows:
            distance = abs(row.base - atom.base)
            same_size = atom.size >= 0.85 * row.size
            limit = max(1.5, 0.35 * row.size) if same_size else max(2.5, 0.9 * row.size)
            if distance <= limit and (best_distance is None or distance < best_distance):
                best, best_distance = row, distance
        if best is None:
            rows.append(Row([atom]))
        else:
            best.atoms.append(atom)
    return rows


def _formula_rows(rows: list[Row], tag_base: float | None) -> list[Row]:
    """Непрерывная полоса строк-выкладок вокруг строки с номером формулы."""
    rows.sort(key=lambda row: row.base)
    flags = [_math_row(row) for row in rows]
    if tag_base is not None and rows:
        # полоса не может уходить далеко от номера: выше на странице стоят
        # другие формулы, и их строки в кроп не должны попадать
        rows = [row for row in rows if tag_base - 62 <= row.base <= tag_base + 12]
        flags = [_math_row(row) for row in rows]
        if not rows:
            return []
        start = min(range(len(rows)), key=lambda i: abs(rows[i].base - tag_base))
    else:
        start = None
    if start is None:
        return [row for row, flag in zip(rows, flags) if flag]
    if not flags[start]:
        # строка номера — отдельный фрагмент (только «(А.6)»), полосу ищем
        # среди соседних математических строк
        flags[start] = True
    low = start
    while low > 0 and flags[low - 1]:
        low -= 1
    high = start
    while high + 1 < len(rows) and flags[high + 1]:
        high += 1
    return rows[low : high + 1]


def _math_row(row: Row) -> bool:
    """Строка — часть выкладки, а не объяснение вокруг неё."""
    atoms = row.atoms
    math = [atom for atom in atoms if atom.math]
    if not math:
        return False
    if len(math) < 0.6 * len(atoms):
        return False
    # одинокая длинная «строка» — объяснение, а не выкладка
    if len(math) == 1 and len(math[0].latex) > 14:
        return False
    return True


def _is_tag(text: str, tag: str) -> bool:
    stripped = text.strip()
    return stripped == f"({tag})" or bool(MATH_TAG.match(stripped))


def _is_prose(text: str) -> bool:
    """Объяснение вокруг формулы: длинное, с русскими словами и пробелами."""
    stripped = text.strip()
    if len(stripped) < 20:
        return False
    if stripped.count(" ") < 2:
        return False
    return bool(PROSE.search(stripped))


def _span_atoms(span: dict, text: str) -> tuple[list[Atom], list[Atom]]:
    """Атомы одного span: SymbolMT раскладывается по глифам, текст — целиком.

    Возвращает (обычные атомы, куски скобок) — скобки собираются отдельно:
    одна высокая скобка рисуется тремя-четырьмя глифами подряд.
    """
    box = span["bbox"]
    base = span["origin"][1]
    size = float(span["size"])
    font = str(span["font"])
    mathish = _is_mathish(text, font)
    atoms: list[Atom] = []
    brackets: list[Atom] = []

    if "Symbol" in font:
        for char in span["chars"]:
            code = _symbol_code(ord(char["c"]))
            char_box = char["bbox"]
            if code in BRACKET_MAP:
                symbol, opening = BRACKET_MAP[code]
                brackets.append(Atom(
                    latex=symbol,
                    x0=char_box[0], x1=char_box[2], y0=char_box[1], y1=char_box[3],
                    base=char["origin"][1], size=size,
                    kind="open" if opening else "close", math=True,
                ))
                continue
            token = SYMBOL_MAP.get(code)
            if token is None:
                continue
            atoms.append(Atom(
                latex=token,
                x0=char_box[0], x1=char_box[2], y0=char_box[1], y1=char_box[3],
                base=char["origin"][1], size=size,
                kind="rel" if token in ("=", r"\le", r"\ge", r"\ne", r"\approx") else "op",
                math=True,
            ))
        return atoms, brackets

    latex = _text_latex(text)
    if latex:
        atoms.append(Atom(
            latex=latex, x0=box[0], x1=box[2], y0=box[1], y1=box[3],
            base=base, size=size,
            kind="rel" if text.strip() in ("=", "≤", "≥", "≠", "≈") else "text",
            math=mathish,
        ))
    return atoms, brackets


def _one_bracket_per_x(brackets: list[Atom]) -> list[Atom]:
    """Из кусков высокой скобки оставляет один: самый верхний по каждому x."""
    chosen: dict[int, Atom] = {}
    for atom in brackets:
        key = round((atom.x0 + atom.x1) / 2)
        current = chosen.get(key)
        if current is None or atom.y0 < current.y0:
            chosen[key] = atom
    result: list[Atom] = []
    for atom in chosen.values():
        command = r"\left" if atom.kind == "open" else r"\right"
        result.append(Atom(
            latex=command + atom.latex,
            x0=atom.x0, x1=atom.x1, y0=atom.y0, y1=atom.y1,
            base=atom.base, size=atom.size, kind=atom.kind, math=True,
        ))
    return result


def _symbol_code(code: int) -> int:
    """Код глифа SymbolMT в ASCII-диапазон: шрифт кладёт знаки в 0xF000..0xF0FF."""
    return code - 0xF000 if 0xF000 <= code <= 0xF0FF else code


def _is_mathish(text: str, font: str) -> bool:
    """Фрагмент похож на выкладку: цифры/операторы, один символ или курсив.

    Проверка нужна, чтобы в формулу не попали обрывки объяснений: конвертер
    режет абзац на куски («ции», «С);», «, (м»), и по длине их от выкладки
    не отличить. Курсив — признак переменной: индексы в СП набраны курсивом.
    """
    stripped = unicodedata.normalize("NFKC", text).strip()
    if not stripped:
        return False
    if re.search(r"\d", stripped) or re.search(r"[=+\-×÷·*/()\[\]|<>≤≥≠≈Σ∑√]", stripped):
        return True
    if len(stripped) == 1:
        return True
    if "Italic" in font and len(stripped) <= 8:
        return True
    # короткий фрагмент без кириллического слова («W,Z», «in», «po»)
    if len(stripped) <= 5 and not re.search(r"[а-яё]{2,}", stripped):
        return True
    return False


def _text_latex(text: str) -> str:
    """Обычный текст PDF -> LaTeX: индексы, функции, кириллица, десятичная запятая."""
    stripped = unicodedata.normalize("NFKC", text).strip()
    if not stripped:
        return ""
    if CYRILLIC.search(stripped):
        # кириллица в формуле — почти всегда индекс или единица измерения
        return r"\text{" + _escape_cyrillic(stripped) + "}"
    lowered = stripped.lower()
    if lowered in FUNCTIONS:
        return FUNCTIONS[lowered]
    escaped = _escape_math(stripped)
    # десятичная запятая: 0,28 -> 0{,}28 (после экранирования, иначе \{
    return re.sub(r"(\d),(\d)", r"\1{,}\2", escaped)


def _escape_cyrillic(text: str) -> str:
    return text.replace("\\", r"\backslash ").replace("{", r"\{").replace("}", r"\}")


def _escape_math(text: str) -> str:
    replacements = {
        "\\": r"\backslash ", "{": r"\{", "}": r"\}", "%": r"\%",
        "&": r"\&", "#": r"\#", "$": r"\$", "_": r"\_",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


# ------------------------------------------------------------------- сборка
def _base_size(atoms: Iterable[Atom]) -> float:
    sizes = sorted(atom.size for atom in atoms)
    return sizes[len(sizes) // 2] if sizes else 0.0


def _compose(
    atoms: list[Atom], bars: list[tuple[float, float, float]], base_size: float
) -> str:
    """Собирает атомы в LaTeX: сначала дроби, потом строки с индексами."""
    if not atoms:
        return ""
    if bars:
        bar = max(bars, key=lambda item: item[1] - item[0])
        bar_x0, bar_x1, bar_y = bar
        rest_bars = [item for item in bars if item is not bar]

        inside = [
            atom for atom in atoms
            if atom.x1 >= bar_x0 - 5 and atom.x0 <= bar_x1 + 5
        ]
        numerator = [atom for atom in inside if atom.y1 <= bar_y + 0.6]
        denominator = [atom for atom in inside if atom.y0 >= bar_y - 0.6]
        if numerator and denominator:
            claimed = {id(atom) for atom in numerator} | {id(atom) for atom in denominator}
            rest = [atom for atom in atoms if id(atom) not in claimed]
            upper_bars = [item for item in rest_bars if item[2] < bar_y]
            lower_bars = [item for item in rest_bars if item[2] > bar_y]
            upper = _compose(numerator, upper_bars, base_size)
            lower = _compose(denominator, lower_bars, base_size)
            frac = Atom(
                latex=rf"\frac{{{upper}}}{{{lower}}}",
                x0=bar_x0, x1=bar_x1, y0=bar_y, y1=bar_y,
                base=bar_y, size=base_size, kind="frac",
            )
            return _compose(rest + [frac], [], base_size)

    return _rows_latex(atoms, base_size)


def _rows_latex(atoms: list[Atom], base_size: float) -> str:
    rows = _rows(atoms)
    rows.sort(key=lambda row: row.base)
    return " ".join(_row_latex(row, base_size) for row in rows if row.atoms)


def _row_latex(row: Row, base_size: float) -> str:
    atoms = sorted(row.atoms, key=lambda item: item.x0)
    row_base = row.base
    main_size, main_height = _dominant(atoms)
    # пары (атом, готовый LaTeX-кусок): индексы приклеиваются к предыдущему
    # куску, поэтому список кусков короче списка атомов — склеивать по zip
    # нельзя, иначе куски съезжают и пропадают пробелы вокруг операторов
    pieces: list[tuple[Atom, str]] = []
    index = 0
    while index < len(atoms):
        atom = atoms[index]
        script = _script_kind(atom, row_base, main_size, main_height)
        if script is None:
            pieces.append((atom, atom.latex))
            index += 1
            continue
        # серия подряд идущих атомов одного знака — один индекс
        group = [atom]
        index += 1
        while index < len(atoms) and _script_kind(
            atoms[index], row_base, main_size, main_height
        ) == script:
            group.append(atoms[index])
            index += 1
        body = _glue(group, main_size, base_size)
        command = "^{" if script == "sup" else "_{"
        if pieces:
            base_atom, previous = pieces[-1]
            pieces[-1] = (base_atom, previous + command + body + "}")
        else:
            pieces.append((atom, command + body + "}"))
    return _glue_pairs(pieces, main_size)


def _glue_pairs(pieces: list[tuple[Atom, str]], row_size: float) -> str:
    """Склеивает готовые куски, вставляя пробел на просвете или после команды."""
    result = ""
    previous: Atom | None = None
    previous_piece = ""
    for atom, piece in pieces:
        if previous is not None:
            gap = atom.x0 - previous.x1
            needs_space = gap > 0.12 * max(row_size, 1.0)
            # \cdotA — не LaTeX: команда без разделителя съедает букву
            if re.search(r"\\[A-Za-z]+$", previous_piece):
                needs_space = True
            if needs_space and not piece.startswith(("^", "_")) and not previous_piece.endswith(("{", "(")):
                result += " "
        result += piece
        previous, previous_piece = atom, piece
    return result


def _script_kind(
    atom: Atom, row_base: float, main_size: float, main_height: float
) -> str | None:
    """sup/sub — если глиф меньше строки и сдвинут по базовой линии.

    Сравнение идёт и по кеглю, и по высоте глифа: у SymbolMT кегль тот же,
    что у букв, а высота глифа — меньше. Дробь всегда стоит в строке как
    обычный атом: она занимает всю высоту и индексов вокруг себя не образует.
    """
    if atom.kind == "frac":
        return None
    height = atom.y1 - atom.y0
    # нужны оба признака: кегль и высота глифа. Одного кегля мало — в строку
    # попадают фрагменты соседнего абзаца того же размера
    small = atom.size <= 0.85 * main_size and height <= 0.85 * main_height
    if not small:
        return None
    delta = atom.base - row_base
    if delta <= -0.8:
        return "sup"
    if delta >= 0.8:
        return "sub"
    return None


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _dominant(atoms: list[Atom]) -> tuple[float, float]:
    """Кегль и высота основы строки: размер, занимающий больше всего ширины.

    Индексы мельче основы, а фрагмент соседнего абзаца хотя и бывает крупнее,
    но занимает мало места — поэтому вес по ширине, а не по количеству.
    """
    weight: dict[float, float] = {}
    for atom in atoms:
        key = round(atom.size, 1)
        weight[key] = weight.get(key, 0.0) + atom.width
    if not weight:
        return 0.0, 1.0
    main_size = max(weight, key=lambda size: weight[size])
    heights = [
        atom.y1 - atom.y0
        for atom in atoms
        if abs(atom.size - main_size) < 0.2 and atom.y1 > atom.y0
    ]
    return main_size, (_median(heights) or 1.0)


def _glue(
    atoms: list[Atom], row_size: float, base_size: float
) -> str:
    """Склеивает атомы индекса, вставляя пробел только на заметном просвете."""
    pieces = [(atom, atom.latex) for atom in atoms]
    return _glue_pairs(pieces, row_size)


def _tidy(latex: str) -> str:
    """Косметика: лишние пробелы, пустые \\text{}, парные \\left\\right."""
    latex = re.sub(r"\s+", " ", latex).strip()
    latex = latex.replace(r"\text{}", "")
    latex = re.sub(r"\\left(.)\s*\\right\1", r"\1", latex)
    # десятичная запятая могла прийти двумя фрагментами («0» и «,28»)
    latex = re.sub(r"(\d)\s*,\s*(\d)", r"\1{,}\2", latex)
    latex = re.sub(r"\s*\{\s*,\s*\}", "{,}", latex)
    # наложения шрифтов дают удвоенные операторы («==», «++»)
    latex = re.sub(r"(?<![\\A-Za-z])([=+\-])\s*\1(?![A-Za-z])", r"\1", latex)
    latex = latex.replace(r"\cdot \cdot", r"\cdot")
    return _balance_brackets(latex).strip()


def _balance_brackets(latex: str) -> str:
    """Убирает \\left/\\right без пары: конвертер часто теряет один из них.

    Пара считается по порядку: \\left открывает, \\right закрывает. Лишние
    команды (и их скобки) выбрасываются — текст внутри остаётся.
    """
    pattern = re.compile(r"\\(?:left|right)([()\[\]{}|])")
    pairs: list[tuple[int, int, str]] = []      # (начало, конец, команда)
    depth = 0
    for match in pattern.finditer(latex):
        if match.group(0).startswith(r"\left"):
            depth += 1
            pairs.append((match.start(), match.end(), "left"))
        else:
            if depth > 0:
                depth -= 1
                pairs.append((match.start(), match.end(), "right"))
            else:
                pairs.append((match.start(), match.end(), "drop"))
    # парные \left без \right — тоже лишние
    unclosed = depth
    result = latex
    for start, end, kind in reversed(pairs):
        if kind == "drop" or (kind == "left" and unclosed > 0):
            if kind == "left":
                unclosed -= 1
            result = result[:start] + result[end:]
    return result
