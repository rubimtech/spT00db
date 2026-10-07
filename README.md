# spT00db — база нормативных документов

Библиотека СП/ГОСТ/СанПиП в SQLite + Qdrant, чтобы агент-проверяющий мог сверять
проектную документацию с нормами: семантический поиск по смыслу, лексический по
формулировкам и точный переход по номеру пункта («п. 7.6.9»).

## Схема

```
PDF (URL, антибот-обход)
  -> pymupdf4llm page_chunks      постраничный Markdown + номер страницы
  -> convert.clean_page           чистка колонтитулов/номеров/отметок редакции
  -> convert.join_pages           склейка страниц: рванные абзацы и таблицы
  -> chunk.chunk_markdown         чанки по ПУНКТАМ нормы (граница = п. 7.6.9, таблица)
  -> SQLite (data/normbase.db)    documents + chunks + FTS5 (BM25)
  -> bge-m3 ONNX (1024d, L2)      ~67 текстов/сек, CPU, локально
  -> Qdrant norm_docs             payload: doc_code, clause_id, page_*, text
```

Почему три слоя поиска:

| Режим | Что даёт | Пример запроса |
|---|---|---|
| `clause` | точная ссылка, порядок документа | `('СП 60.13330.2020', '7.6.9')` |
| `keyword` | BM25 по FTS5 — точные формулировки | «предел огнестойкости воздуховодов» |
| `semantic` | bge-m3 — синонимы, перефразировки | «куда выбрасывать воздух из вентиляции» |
| `hybrid` | RRF-слияние (по умолчанию) | свободный текст вопроса |

## Структура

```
src/norms_base/
  config.py      пути/константы (DB_PATH, QDRANT_URL, COLLECTION=norm_docs, размеры чанков)
  download.py    fetch_pdf: cookies корня + Referer + Sec-Fetch-* -> обход антибота
  convert.py     pdf_to_markdown: построчная чистка, карта (offset -> страница)
  chunk.py       chunk_markdown: границы по пунктам/таблицам, build_embed_text
  embed.py       Embedder: bge-m3 ONNX, ленивая сессия, L2-нормализация
  storage.py     Store: SQLite schema, FTS5, find_clause/search_fts
  vectorstore.py VectorStore: Qdrant, payload-индексы, point_id = uuid5(code:ord)
  pipeline.py    add_document(): PDF -> ... -> Qdrant, reembed_pending(), status()
  search.py      API для агента: clause/section/keyword/semantic/hybrid/docs
data/            (в .gitignore) normbase.db, pdf/, md/ — артефакты, восстановимы
```

## Статус (2026-10-07)

**Сделано**

- Модули `convert/chunk/storage/vectorstore/search/pipeline` написаны, импортируются.
- Эмбеддер проверен на живой модели: 1024d, L2=1.0, batch16 = 0.21 с (~67 текстов/сек).
- Скачиватель проверен вручную на `nav.tn.ru`: прямой GET — заглушка 1295 байт
  «Forbidden»; сессия (GET корня -> cookies) + Referer -> 200 application/pdf, 109 стр.
- Конверсия проверена на СП 60.13330.2020: 346 пунктов из 346 сохранено,
  13 приложений (А–П), 273 строки таблиц, ноль колонтитулов и отметок редакции.
- Референс-образцы чистки (чего ради всё это): колонтитул вплетён в таблицу,
  абзац, разорванный границей страницы, повторённая шапка таблицы на новой странице.

**Не сделено (следующая сессия)**

1. `norms.py` — CLI (`add/search/clause/docs/status/reembed`, флаг `--json` для агента). *Блокер для использования.*
2. Прогон `add_document` по СП 60.13330.2020 end-to-end + сверка `status()` (delta = 0).
3. Тесты (smoke: конверсия фикстуры -> чанки -> FTS-запрос -> вектор в тестовой коллекции).
4. Проверка качества выдачи на 5-10 заведомых запросах (калибровка `score_threshold`).
5. README-раздел «как подключать следующий документ» после первых 2-3 инжестов.

## Грабли, на которые уже наступили

- **`web_extract` (Firecrawl) искажает русский текст** этого PDF («отоления,
  ventilляции», «жилинно-коммунального») — локальная экстракция через pymupdf4llm
  даёт чистый текст. Не использовать Firecrawl как источник текста нормативов.
- **pymupdf4llm**: `page_chunks=True` -> элементы с `metadata.page_number` и `text`;
  в доках сигнатура обёртки `(*args, **kwargs)`, поэтому `inspect.signature` не поможет.
- **Субпроцессы pymupdf4llm** кидают `UnicodeDecodeError` в потоках чтения stdout
  (кодировка CP1251 у вывода) — не влияет на результат, но лог пугает.
- **Qdrant**: чтение payload — `POST /collections/{c}/points/scroll` (GET отвергается).
- **`execute_code` (ядро Hermes) — другой интерпретатор**, в нём нет
  pymupdf4llm/qdrant_client/onnxruntime; запускать код проекта через
  `C:/Users/olegs/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe`.
- **Ставить пакеты** в этот venv: `uv pip install --python "...venv/Scripts/python.exe" <pkg>`.

## Источник первого документа

- СП 60.13330.2020 «Отопление, вентиляция и кондиционирование воздуха» (проект, 2-я редакция, 109 стр.)
  `https://nav.tn.ru/cloud/iblock/d02/d024f677f0b3a18dc7d4288ee664b4f7/SP-60.13330.2020-Otoplenie_-ventilyatsiya-i-konditsionirovanie-vozdukha.pdf`

## Запуск

```bash
PY="C:/Users/olegs/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe"
# до появления CLI:
"$PY" -c "import sys; sys.path.insert(0, 'src'); from norms_base.pipeline import add_document; \
add_document('СП 60.13330.2020', 'Отопление, вентиляция и кондиционирование воздуха', \
             url='https://nav.tn.ru/.../SP-60.13330.2020-....pdf')"
```
