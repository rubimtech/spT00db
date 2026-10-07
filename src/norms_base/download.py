"""Скачивание нормативных PDF, устойчивое к антибот-защите хостинга.

Проверено на nav.tn.ru (2026-10-07): прямой GET из curl отдаёт HTML-заглушку
1295 байт «Forbidden / If you are not a bot» — даже с Chrome User-Agent.
Рабочая схема:

1. GET корня домена в той же сессии — получаем cookies (обычно 200);
2. GET файла тем же cookie-джаром, с ``Referer`` на корень и заголовками
   ``Sec-Fetch-*`` как у настоящего браузера -> 200 application/pdf.

Файл считается скачанным только по магии ``%PDF-``: заглушку ошибки
принимать за PDF нельзя — иначе дальше «конвертируется» мусор.

Челлендж эпизодический (≈1 запрос из 3, замер 2026-10-07): ServicePipe
отдаёт HTML со скриптом servicepipe.tech, но в ТОМ ЖЕ ответе ставит
``spid``/``spst`` — поэтому повтор делается в той же сессии (:func:`fetch_pdf`),
а не в новой.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import SplitResult, urlsplit, urlunsplit

import requests

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


class DownloadError(RuntimeError):
    """Скачивание не удалось или сервер отдал не PDF."""


@dataclass(slots=True)
class FetchResult:
    path: Path
    sha256: str
    size: int
    url: str
    content_type: str
    from_cache: bool


def fetch_pdf(
    url: str,
    dest_dir: Path,
    *,
    filename: str | None = None,
    timeout: int = 90,
    attempts: int = 5,
    delay: float = 2.0,
    verbose: bool = False,
) -> FetchResult:
    """Скачивает PDF в ``dest_dir``, проходя антибот-челлендж повтором в ТОЙ ЖЕ сессии.

    Хостинг защищён ServicePipe: примерно в одном запросе из трёх вместо файла
    приходит HTML-челлендж (1892 байта, скрипт servicepipe.tech). Ключевая
    деталь — **сам ответ с челлендже выставляет cookies** ``spid``/``spst``,
    после чего следующий запрос того же инстанса проходит. Поэтому повтор в
    НОВОЙ сессии бессмыслен: она стартует без этих cookies и получает
    челлендж снова — ровно эта ошибка и была сделана ранее.

    Файл принимается только по магии ``%PDF-``: заглушку ошибки дальше
    «конвертировать» нельзя.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise DownloadError(f"неподдерживаемая схема URL: {url!r}")
    origin = urlunsplit((parts.scheme, parts.netloc, "/", "", ""))

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        }
    )
    last_error: DownloadError | None = None
    try:
        session.get(origin, timeout=timeout)  # Шаг 1: cookies корневой страницы
        for attempt in range(1, max(attempts, 1) + 1):
            # Шаг 2: файл тем же cookie-джаром, с Referer и Sec-Fetch-* как у браузера
            response = session.get(url, headers=_file_headers(origin), timeout=timeout)
            body = response.content
            if response.status_code == 200 and body.startswith(b"%PDF-"):
                return _store(dest_dir, body, url, parts, filename, response)
            last_error = _problem(url, response, body)
            if verbose and attempt < attempts:
                print(
                    f"[pdf] попытка {attempt}/{attempts}: {last_error}. "
                    f"Повтор через {delay * attempt:.0f} с в той же сессии.",
                    flush=True,
                )
            if attempt < attempts:
                time.sleep(delay * attempt)
    except requests.RequestException as exc:
        raise DownloadError(f"сетевая ошибка при скачивании {url}: {exc}") from exc

    raise last_error if last_error else DownloadError(f"не удалось скачать {url}")


def _file_headers(origin: str) -> dict[str, str]:
    return {
        "Referer": origin,
        "Accept": "application/pdf,application/xhtml+xml,text/html;q=0.9,*/*;q=0.8",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Upgrade-Insecure-Requests": "1",
    }


def _problem(url: str, response: requests.Response, body: bytes) -> DownloadError:
    if response.status_code != 200:
        return DownloadError(f"HTTP {response.status_code} для {url}")
    preview = body[:300].decode("utf-8", errors="replace")
    # метки ServicePipe живут дальше 300 символов — ищем по первым КБ
    head = body[:4000].decode("utf-8", errors="replace")
    if "servicepipe" in head or "js-challenge" in head:
        return DownloadError("антибот-челлендж ServicePipe вместо PDF")
    return DownloadError(
        "сервер отдал не PDF (защита от ботов или редирект на страницу ошибки). "
        f"Начало ответа: {preview!r}"
    )


def _store(
    dest_dir: Path,
    body: bytes,
    url: str,
    parts: SplitResult,
    filename: str | None,
    response: requests.Response,
) -> FetchResult:
    digest = hashlib.sha256(body).hexdigest()
    name = filename or _suggest_name(parts.path, digest)
    path = dest_dir / name
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        return FetchResult(path, digest, len(body), url, "application/pdf", True)
    path.write_bytes(body)
    time.sleep(0.5)  # вежливость к хостингу перед следующим скачиванием
    return FetchResult(
        path=path,
        sha256=digest,
        size=len(body),
        url=url,
        content_type=response.headers.get("Content-Type", "application/pdf"),
        from_cache=False,
    )


def _suggest_name(path_part: str, digest: str) -> str:
    """Имя файла из URL + короткий хэш: одинаковые URL на разных зеркалах не слипаются."""
    raw = Path(path_part).name or "document.pdf"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._") or "document"
    if not stem.lower().endswith(".pdf"):
        stem += ".pdf"
    return f"{digest[:10]}_{stem}"
