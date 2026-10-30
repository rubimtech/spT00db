"""Картинки из PDF: извлечение, хранение, описание.

Задачи модуля:

1. *Найти* в PDF картинки: растровые вставки (``page.get_images``) и
   векторные схемы/графики — по подписи вида «Рисунок N» (кроп страницы
   над подписью). Всё складывается в ``data/img/<код документа>/``.
2. *Хранить* найденное в таблице ``figures`` SQLite: страница, подпись,
   путь к файлу, описание.
3. *Описать* — двумя способами, выбирается в ``figures.yaml``:

   * ``vision.model`` указана → ``norms.py figures --auto`` шлёт картинку
     во внешнюю LLM (OpenAI-совместимый API): base_url и ключ берутся из
     Hermes ``config.yaml`` (провайдер ``auxiliary.vision``), пока не
     переопределены явно в ``figures.yaml``;
   * ``vision.model`` пустая (по умолчанию) → описывает агент: смотрит
     файлы (``norms.py figures`` печатает пути) и заносит текст командой
     ``norms.py describe <id> --text "…"``.

Описание живёт в двух местах: строка ``figures.description`` — источник
правды, она переживает переиндексацию документа; и метка внутри текста
чанка (``<!-- fig:N -->…<!-- /fig:N -->``) — так описание автоматически
попадает в FTS5 и в эмбеддинг и находится поиском вместе с пунктом,
рядом с которым картинка стоит.
"""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import FIGURES_YAML, REPO_ROOT
from . import config as _config  # IMG_DIR читаем динамически: тесты его патчат
from .storage import Store

#: Подпись картинки в начале строки PDF: «Рисунок Ж.1…», «Рисунок 2 – …»,
#: «Схема 3…», «График 1…». Цифра или номер приложения («Ж.1») обязателен —
#: иначе слово «рисунок» внутри обычного предложения дало бы ложный кроп.
CAPTION_RE = re.compile(
    r"^\s*(?:Рисунок|Рис\.|Схема|График|Чертёж|Чертеж)\s*[-–—.]?\s*(?:\d+|[А-Я]\.\d+)",
    re.IGNORECASE,
)

#: Максимальная высота кропа векторной схемы: выше подписи искать не дальше.
MAX_CROP_HEIGHT = 620.0

#: Системный промпт для автоописания: строго факты с картинки, без домыслов.
VISION_PROMPT = (
    "Ты описываешь изображения из нормативного документа по строительству "
    "(СП/ГОСТ/СанПиН) для базы, по которой потом ищут текстом. "
    "Опиши картинку 3-6 предложениями на русском: тип (схема, график, "
    "чертёж, таблица-рисунок, фото), что изображено, обозначения и числа "
    "из подписи/легенды, какой вывод делает по этой картинке документ. "
    "Только то, что реально видно на изображении, без домыслов."
)
#: Описание короче этого — модель вернула мусор («Рисун»), а не описание.
MIN_DESC_CHARS = 80

@dataclass(slots=True)
class FigureCandidate:
    """Картинка, найденная в PDF, ещё не записанная в базу."""

    page: int          # номер страницы, с 1
    kind: str          # raster | crop
    xref: int | None   # xref растровой вставки, None для кропа
    caption: str       # подпись «Рисунок N…» или ""
    bbox: tuple[float, float, float, float] | None
    path: Path         # куда сохранён файл


@dataclass(slots=True)
class VisionConfig:
    """Подключение к внешней LLM для автоописания (figures.yaml)."""

    model: str = ""
    base_url: str = ""
    api_key: str = ""
    provider: str = ""
    max_dim: int = 1280
    timeout: float = 120.0
    max_tokens: int = 2000

    @property
    def auto(self) -> bool:
        """True — можно описывать автоматически (модель и доступ заданы)."""
        return bool(self.model and self.base_url and self.api_key)


# --------------------------------------------------------------- извлечение
def extract_figures(pdf_path: Path | str, doc_code: str, img_dir: Path | str | None = None) -> list[FigureCandidate]:
    """Вырезает картинки из PDF в ``img_dir`` (по умолчанию ``data/img/<slug>``)."""
    import pymupdf

    img_dir = Path(img_dir) if img_dir else _config.IMG_DIR / _slug(doc_code)
    img_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(str(pdf_path))
    found: list[FigureCandidate] = []
    try:
        for pno in range(len(doc)):
            page = doc[pno]
            captions = _captions(page)
            raster_rects: list[Any] = []

            # 1) растровые вставки
            for xref in sorted({image[0] for image in page.get_images(full=True)}):
                pix = pymupdf.Pixmap(doc, xref)
                if pix.colorspace and pix.colorspace.n > 3:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                if pix.width < 40 or pix.height < 40:
                    continue  # иконки/буллиты — не картинки норматива
                path = img_dir / f"p{pno + 1:03d}_x{xref}.png"
                pix.save(str(path))
                rect = _image_rect(page, xref)
                if rect is not None:
                    raster_rects.append(rect)
                found.append(
                    FigureCandidate(
                        page=pno + 1,
                        kind="raster",
                        xref=xref,
                        caption=_nearest_caption(captions, rect),
                        bbox=tuple(rect) if rect is not None else None,
                        path=path,
                    )
                )

            # 2) векторные схемы: кроп страницы над подписью
            crop_index = 0
            for cap_text, cap_rect in captions:
                clip = _figure_clip(page, cap_rect)
                if clip is None:
                    continue
                if any(rect.intersects(clip) for rect in raster_rects):
                    continue  # подпись уже покрыта растровой картинкой выше
                crop_index += 1
                path = img_dir / f"p{pno + 1:03d}_crop{crop_index}.png"
                page.get_pixmap(clip=clip, dpi=200).save(str(path))
                found.append(
                    FigureCandidate(
                        page=pno + 1,
                        kind="crop",
                        xref=None,
                        caption=cap_text,
                        bbox=(clip.x0, clip.y0, clip.x1, clip.y1),
                        path=path,
                    )
                )
    finally:
        doc.close()
    return found


def sync_figures(
    store: Store,
    doc_id: int,
    pdf_path: Path | str,
    doc_code: str,
    *,
    verbose: bool = True,
) -> int:
    """Пересыпает картинки PDF в таблицу ``figures`` и возвращает их число.

    Идемпотентно: путь картинки уникален в пределах документа, повторный
    вызов не плодит дубли и не трогает уже записанные описания. После
    синхронизации описания, пережившие переиндексацию, заново вшиваются
    в текст чанков (``reapply_figure_notes``).
    """
    candidates = extract_figures(pdf_path, doc_code)
    for fig in candidates:
        store.upsert_figure(
            doc_id=doc_id,
            page=fig.page,
            kind=fig.kind,
            xref=fig.xref,
            caption=fig.caption,
            bbox=",".join(str(value) for value in fig.bbox) if fig.bbox else "",
            path=str(fig.path),
        )
    described = store.reapply_figure_notes(doc_id)
    store.commit()
    if verbose and candidates:
        img_dir = _config.IMG_DIR / _slug(doc_code)
        print(f"[img] {len(candidates)} изображений, {described} с описанием -> {img_dir}", flush=True)
    return len(candidates)


# ------------------------------------------------------------------- конфиг
def load_vision_config(path: Path | str | None = None) -> VisionConfig:
    """Читает ``figures.yaml`` и дополняет base_url/key из Hermes config.yaml.

    Пустые поля ``figures.yaml`` — не ошибка, а «возьми из Hermes»: провайдер
    берётся из ``auxiliary.vision`` (а при его отсутствии — из ``model``),
    ключ — из env-переменной ``key_env`` этого провайдера. ``NORMS_VISION_*``
    переопределяют всё (удобно для разовой проверки без правки файла).
    """
    import yaml

    cfg_path = Path(path) if path else FIGURES_YAML
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        raw = data.get("vision") or {}

    cfg = VisionConfig(
        model=str(raw.get("model") or ""),
        provider=str(raw.get("provider") or ""),
        base_url=str(raw.get("base_url") or ""),
        max_dim=int(raw.get("max_dim") or 1280),
        timeout=float(raw.get("timeout") or 120.0),
        max_tokens=int(raw.get("max_tokens") or 2000),
    )
    # ключ: env NORMS_VISION_API_KEY > api_key литерально из yaml >
    # env(api_key_env из yaml) > key_env провайдера из Hermes config.yaml
    cfg.api_key = str(raw.get("api_key") or "")
    cfg.model = os.environ.get("NORMS_VISION_MODEL", cfg.model)
    cfg.base_url = os.environ.get("NORMS_VISION_BASE_URL", cfg.base_url)

    key_env = os.environ.get("NORMS_VISION_API_KEY_ENV") or str(raw.get("api_key_env") or "")
    if key_env and not cfg.api_key:
        cfg.api_key = os.environ.get(key_env) or ""
    cfg.api_key = os.environ.get("NORMS_VISION_API_KEY") or cfg.api_key

    if cfg.model and (not cfg.base_url or not cfg.api_key):
        hermes = _hermes_config()
        vision = ((hermes.get("auxiliary") or {}).get("vision")) or {}
        # auxiliary.vision.provider может быть "auto" (автовыбор самого Hermes) —
        # для прямых API-вызовов это не имя провайдера, пропускаем до model.provider
        candidates = (
            cfg.provider,
            str(vision.get("provider") or ""),
            str((hermes.get("model") or {}).get("provider") or ""),
        )
        provider = next((item for item in candidates if item and item != "auto"), "")
        cfg.provider = provider
        prov = (hermes.get("providers") or {}).get(provider) or {}
        if not cfg.base_url:
            cfg.base_url = str(prov.get("base_url") or vision.get("base_url") or "")
        if not cfg.api_key and prov.get("key_env"):
            cfg.api_key = os.environ.get(str(prov["key_env"]), "")
    return cfg


def _hermes_config_path() -> Path:
    explicit = os.environ.get("HERMES_CONFIG")
    if explicit:
        return Path(explicit)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / "hermes" / "config.yaml"
    return Path.home() / "AppData" / "Local" / "hermes" / "config.yaml"


def _hermes_config() -> dict[str, Any]:
    path = _hermes_config_path()
    if not path.exists():
        return {}
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# ------------------------------------------------------------------ LLM API
def describe_with_llm(image_path: Path | str, caption: str, cfg: VisionConfig) -> str:
    """Одно изображение → описание через OpenAI-совместимый chat/completions.

    Reasoning-модели иногда съедают весь max_tokens на размышления и
    возвращают пустой content — тогда запрос повторяется с двойным бюджетом.
    """
    import httpx

    if not cfg.auto:
        raise ValueError("vision.model/base_url/api_key не заданы — автоописание выключено")
    png = _png_bytes(Path(image_path), cfg.max_dim)
    b64 = base64.b64encode(png).decode("ascii")
    last_error = "нет попыток"
    for max_tokens in (cfg.max_tokens, cfg.max_tokens * 2):
        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": VISION_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Подпись в документе: {caption or 'подписи нет'}"},
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
        text = _parse_description(response.json(), cfg.model)
        if text and len(text) >= MIN_DESC_CHARS:
            return text
        last_error = "пустой или слишком короткий content (размышления/обрезка)"
    raise ValueError(f"LLM не вернула описание (модель {cfg.model}): {last_error}")


def _parse_description(data: dict, model: str) -> str:
    """Текст описания из ответа chat/completions; пустая строка, если его нет."""
    message = ((data.get("choices") or [{}])[0].get("message")) or {}
    content = message.get("content")
    if content is None:
        content = message.get("text") or ""
    if isinstance(content, list):  # провайдеры отдают список частей
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    text = str(content or "").strip()
    return "" if not text or text.lower() == "none" else text


def _png_bytes(path: Path, max_dim: int) -> bytes:
    """PNG для запроса: сжат pymupdf до max_dim по длинной стороне."""
    import pymupdf

    pix = pymupdf.Pixmap(str(path))
    while max(pix.width, pix.height) > max_dim and min(pix.width, pix.height) > 1:
        if hasattr(pix, "shrink"):
            pix.shrink(1)
        else:  # pragma: no cover - крайний случай старого pymupdf
            break
    return pix.tobytes("png")


# ------------------------------------------------------------------ internal
def _captions(page: Any) -> list[tuple[str, Any]]:
    """Строки-подписи страницы: (текст, rect)."""
    import pymupdf

    captions: list[tuple[str, Any]] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if text and CAPTION_RE.match(text):
                captions.append((text, pymupdf.Rect(line["bbox"])))
    return captions


def _nearest_caption(captions: list[tuple[str, Any]], rect: Any) -> str:
    """Подпись, ближайшая к картинке (по вертикали); иначе первая на странице."""
    if not captions:
        return ""
    if rect is None:
        return captions[0][0]

    def distance(item: tuple[str, Any]) -> float:
        _, cap = item
        if cap.y0 >= rect.y1:
            return cap.y0 - rect.y1
        if cap.y1 <= rect.y0:
            return rect.y0 - cap.y1
        return 0.0  # подпись внутри/пересекается с картинкой

    return min(captions, key=distance)[0]


def _image_rect(page: Any, xref: int) -> Any:
    try:
        rects = page.get_image_rects(xref)
    except Exception:  # noqa: BLE001 - позиция не критична, нужна только подпись
        return None
    return rects[0] if rects else None


def _figure_clip(page: Any, cap_rect: Any) -> Any | None:
    """Прямоугольник кропа: графика страницы над подписью + сама подпись.

    Пусто (None), если над подписью нет ни одной векторной графики — такая
    подписи соответствует либо растровая картинка (сохранена отдельно), либо
    картинки в документе нет.
    """
    import pymupdf

    page_rect = page.rect
    band = pymupdf.Rect(
        page_rect.x0 + 30,
        max(page_rect.y0 + 40, cap_rect.y0 - MAX_CROP_HEIGHT),
        page_rect.x1 - 30,
        min(page_rect.y1 - 20, cap_rect.y1 + 6),
    )
    if band.width < 50 or band.height < 40:
        return None

    rects = [pymupdf.Rect(item["rect"]) for item in page.get_drawings() if pymupdf.Rect(item["rect"]).intersects(band)]
    if not rects:
        return None

    union = pymupdf.Rect(rects[0])
    for rect in rects[1:]:
        union.x0 = min(union.x0, rect.x0)
        union.y0 = min(union.y0, rect.y0)
        union.x1 = max(union.x1, rect.x1)
        union.y1 = max(union.y1, rect.y1)
    # пересечение с полосой поиска
    union = pymupdf.Rect(
        max(union.x0, band.x0),
        max(union.y0, band.y0),
        min(union.x1, band.x1),
        min(union.y1, band.y1),
    )
    if union.width < 60 or union.height < 40:
        union = band  # крохотные штрихи — берём полосу целиком

    # подпись обязана попасть в кроп: по ней описание привязывается к пункту
    return pymupdf.Rect(
        min(union.x0, cap_rect.x0),
        union.y0,
        max(union.x1, cap_rect.x1),
        min(band.y1, cap_rect.y1 + 6),
    )


def _slug(code: str) -> str:
    return re.sub(r"[^\w.]+", "_", code, flags=re.UNICODE).strip("_") or "document"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
