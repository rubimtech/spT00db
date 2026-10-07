"""Эмбеддинг нормативных текстов: BAAI/bge-m3 в ONNX (1024d, Cosine, L2).

Почему bge-m3, а не intfloat/multilingual-e5-large из price_parser:

* e5 требует асимметричных префиксов ``passage:``/``query:`` — их теряют при
  написании нового поисковика, а потерянный префикс молча портит выдачу
  (в price_parser это зафиксировано в SPEC §10.1 именно как риск);
* у bge-m3 контекст 8192 токена — пункт вместе с таблицей помещается целиком;
* веса уже локально лежат в ``models/bge-m3-onnx``: их же использует
  memory-store MCP и ws-* коллекции, поэтому агент-проверяющий может
  эмбеддить запросы той же моделью, что и индекс.

Пуллинг внутри графа (выход ``sentence_embedding``), наружу нормализуется
только L2 — для Cosine это обязательное условие, иначе сравнение бессмысленно.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import BGE_M3_DIR, DIM, EMBED_THREADS, MAX_TOKENS

_WS_LIMIT = 50_000  # обрезка до эмбеддинга: длиннее всё равно не влезет в модель


class Embedder:
    """Ленивая обёртка над onnxruntime: сессия поднимается при первом вызове.

    Один экземпляр = одна сессия. Сессия тяжёлая (2.2 ГБ весов на диске,
    ~8 с на прогрев), поэтому её держат между батчами.
    """

    def __init__(self, model_dir: Path | str | None = None, threads: int | None = None) -> None:
        self.model_dir = Path(model_dir or BGE_M3_DIR)
        # 8 потоков: замер 2026-10-07 (4 -> 8 = +25% к скорости на длинных чанках)
        self.threads = int(threads if threads is not None else EMBED_THREADS)
        self._session = None
        self._tokenizer = None
        self._pad_id = 0

    # ------------------------------------------------------------------ lazy

    def _ensure(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort
        import tokenizers

        model = self.model_dir / "model.onnx"
        tokenizer_path = self.model_dir / "tokenizer.json"
        for path in (model, self.model_dir / "model.onnx_data", tokenizer_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"модель bge-m3 не найдена: {path}. Задайте NORMS_BGE_M3 "
                    f"или положите экспорт BAAI/bge-m3 в {self.model_dir}"
                )
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        self._session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        self._tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        # XLM-R: pad = <s> (id 0); tokenizer отдаёт явный [PAD] только если он есть.
        self._pad_id = self._tokenizer.token_to_id("[PAD]") or 0

    @property
    def dim(self) -> int:
        return DIM

    @property
    def loaded(self) -> bool:
        return self._session is not None

    # ---------------------------------------------------------------- embed

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Эмбеддит батч -> (n, 1024) float32 с единичной L2-нормой."""
        items = [str(text).replace("\x00", " ")[:_WS_LIMIT] for text in texts]
        if not items:
            return np.zeros((0, DIM), dtype=np.float32)
        self._ensure()
        encodings = self._tokenizer.encode_batch(items)
        cropped = [enc.ids[:MAX_TOKENS] for enc in encodings]
        width = max(len(ids) for ids in cropped)
        input_ids = np.full((len(cropped), width), self._pad_id, dtype=np.int64)
        attention = np.zeros((len(cropped), width), dtype=np.int64)
        for row, (enc, ids) in enumerate(zip(encodings, cropped, strict=True)):
            input_ids[row, : len(ids)] = ids
            attention[row, : len(ids)] = enc.attention_mask[:MAX_TOKENS]
        names = [out.name for out in self._session.get_outputs()]
        outputs = self._session.run(None, {"input_ids": input_ids, "attention_mask": attention})
        if "sentence_embedding" in names:
            vectors = outputs[names.index("sentence_embedding")]
        else:  # pragma: no cover - запасной путь, если экспорт без пулинга
            vectors = outputs[0]
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))

    def embed_one(self, text: str) -> np.ndarray:
        """Эмбеддинг одного запроса -> (1024,) float32."""
        return self.embed([text])[0]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, np.float32(1.0), norms)
    return (vectors / norms).astype(np.float32, copy=False)


def embed_texts(
    texts: Sequence[str],
    batch_size: int = 64,
    embedder: Embedder | None = None,
    progress_every: int = 0,
) -> np.ndarray:
    """Батчинг с прогрессом: ~60-70 текстов/сек на 4 потока CPU (замер 2026-10-07)."""
    engine = embedder or Embedder()
    items = list(texts)
    if not items:
        return np.zeros((0, DIM), dtype=np.float32)
    started = time.perf_counter()
    parts: list[np.ndarray] = []
    for start in range(0, len(items), max(int(batch_size), 1)):
        parts.append(engine.embed(items[start : start + batch_size]))
        if progress_every and (start // batch_size) % int(progress_every) == 0:
            elapsed = max(time.perf_counter() - started, 1e-9)
            done = min(start + batch_size, len(items))
            print(f"[embed] {done}/{len(items)} | {done / elapsed:.1f} текста/сек", flush=True)
    return np.vstack(parts)
