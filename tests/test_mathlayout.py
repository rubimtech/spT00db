"""Восстановление LaTeX по геометрии символов (mathlayout).

Проверяется то, что конвертер теряет: дробь собирается по черте из
``get_drawings``, степень и индекс — по кеглю и смещению базовой линии,
глифы SymbolMT (0xF0xx) переводятся в команды LaTeX. PDF строится на месте:
в текстовом слое формула лежит «плоско», структуру дают только координаты.
"""

from __future__ import annotations

import pymupdf
import pytest

from conftest import FONT
from norms_base.mathlayout import latex_from_region


def _page() -> tuple[pymupdf.Document, pymupdf.Page]:
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_font(fontname="arial", fontfile=str(FONT))
    return doc, page


def _text(page, x: float, y: float, text: str, size: float = 10, font: str = "arial") -> None:
    page.insert_text((x, y), text, fontname=font, fontsize=size, fontfile=str(FONT))


def test_fraction_from_drawing_bar():
    """Дробь: числитель над чертой, знаменатель под ней, номер справа."""
    doc, page = _page()
    _text(page, 100, 300, "353", size=10)
    page.draw_line(pymupdf.Point(98, 304), pymupdf.Point(120, 304), width=0.5)
    _text(page, 96, 318, "273+t", size=10)
    _text(page, 470, 318, "(А.6)", size=9)
    clip = pymupdf.Rect(90, 285, 500, 325)

    latex = latex_from_region(page, clip, "А.6")
    doc.close()
    assert r"\frac{353}{273+t}" in latex, latex


def test_superscript_and_subscript_by_size_and_baseline():
    """Степень поднята и мельче, индекс опущен: оба становятся ^{} и _{}."""
    doc, page = _page()
    _text(page, 100, 300, "ab", size=10)
    _text(page, 113, 294, "2", size=7)          # степень: выше и мельче
    _text(page, 116, 304, "н", size=7)          # индекс: ниже и мельче
    _text(page, 470, 300, "(1.1)", size=9)
    clip = pymupdf.Rect(90, 280, 500, 320)

    latex = latex_from_region(page, clip, "1.1")
    doc.close()
    assert "ab" in latex and "^{2}" in latex and "_{" in latex, latex


def test_symbol_font_glyphs_become_latex_commands():
    """Знаки из SymbolMT (0xF0xx) переводятся в команды, а не в PUA-мусор."""
    doc, page = _page()
    page.insert_font(fontname="symbol", fontfile=r"C:\Windows\Fonts\times.ttf")
    _text(page, 100, 300, "a", size=10)
    page.insert_text(
        (110, 300), "\uf0d7b", fontname="symbol", fontsize=10,
    )
    _text(page, 470, 300, "(2.2)", size=9)
    clip = pymupdf.Rect(90, 280, 500, 320)

    latex = latex_from_region(page, clip, "2.2")
    doc.close()
    assert any(0xE000 <= ord(char) <= 0xF8FF for char in latex) is False, latex


@pytest.mark.parametrize("tag", ["А.6", None])
def test_region_without_math_returns_empty(tag):
    """Строка-ссылка без выкладки: LaTeX не выдумывается."""
    doc, page = _page()
    _text(page, 100, 300, "см. приложение А настоящего свода правил", size=10)
    clip = pymupdf.Rect(90, 280, 500, 320)

    latex = latex_from_region(page, clip, tag)
    doc.close()
    assert latex == ""
