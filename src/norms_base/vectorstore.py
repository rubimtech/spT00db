"""Векторный индекс Qdrant: коллекция ``norm_docs`` (1024d, Cosine, bge-m3).

Решения, продиктованные назначением базы (проверка проектки):

* одна коллекция на всю библиотеку нормативов, а не коллекция на документ:
  запрос про «огнестойкость воздуховодов» обязан пересекать СП, ГОСТ и СанПиН,
  а фильтр по документу ставится опционально на уровне ``doc_code``;
* ``point_id = uuid5(doc_code + ord)`` — переиндексация перезаписывает точку,
  а не плодит дубли (иначе старые чанки отвечают на запросы после правок);
* payload несёт ссылку на SQLite (``chunk_id``) и всю выдачу (пункт, страница,
  раздел): агент-проверяющий получает готовую цитату без второго запроса;
* ``text`` индексируется как TEXT — это даёт лексический match_text внутри
  Qdrant задаром, когда SQLite недоступен.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .config import COLLECTION, DIM, POINT_NAMESPACE, QDRANT_URL

log = logging.getLogger(__name__)

UPSERT_BATCH = 512
SCROLL_BATCH = 256
_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, POINT_NAMESPACE)

PAYLOAD_INDEXES: dict[str, Any] = {}


def _build_payload_indexes() -> dict[str, Any]:
    from qdrant_client import models

    return {
        "doc_code": models.PayloadSchemaType.KEYWORD,
        "clause_id": models.PayloadSchemaType.KEYWORD,
        "section": models.PayloadSchemaType.KEYWORD,
        "kind": models.PayloadSchemaType.KEYWORD,
        "doc_title": models.PayloadSchemaType.KEYWORD,
        "text": models.PayloadSchemaType.TEXT,
        "page_start": models.PayloadSchemaType.INTEGER,
    }


@dataclass(slots=True)
class Hit:
    id: str
    score: float
    payload: dict[str, Any]


class VectorStore:
    """Ленивая обёртка над QdrantClient: на импорте сети нет."""

    def __init__(
        self,
        url: str = QDRANT_URL,
        collection: str = COLLECTION,
        dim: int = DIM,
        timeout: int = 120,
    ) -> None:
        self.url = url
        self.collection = collection
        self.dim = int(dim)
        self.timeout = timeout
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.url, timeout=self.timeout)
        return self._client

    def ensure_collection(self, recreate: bool = False) -> bool:
        """Создаёт коллекцию + payload-индексы; True, если коллекция создана заново."""
        from qdrant_client import models

        client = self.client
        exists = bool(client.collection_exists(self.collection))
        if exists and recreate:
            client.delete_collection(self.collection)
            exists = False
        if not exists:
            client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dim, distance=models.Distance.COSINE, on_disk=True
                ),
            )
        self._ensure_payload_indexes()
        return not exists

    def _ensure_payload_indexes(self) -> list[str]:
        from qdrant_client import models

        global PAYLOAD_INDEXES
        if not PAYLOAD_INDEXES:
            PAYLOAD_INDEXES = _build_payload_indexes()
        try:
            existing = set((self.client.get_collection(self.collection).payload_schema or {}).keys())
        except Exception:  # noqa: BLE001
            existing = set()
        failed: list[str] = []
        for field, schema in PAYLOAD_INDEXES.items():
            if field in existing:
                continue
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema,
                    wait=True,
                )
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{field}: {type(exc).__name__}")
        if failed:
            log.warning("payload-индексы не созданы: %s", "; ".join(failed))
        return failed

    def upsert(self, points: Sequence[Mapping[str, Any]], batch_size: int = UPSERT_BATCH) -> int:
        """Точки ``{id, vector, payload}`` батчами по 512 с wait=True."""
        from qdrant_client import models

        prepared = []
        for point in points:
            vector = np.asarray(point["vector"], dtype=np.float32).ravel()
            if vector.size != self.dim:
                raise ValueError(f"вектор {vector.size} != {self.dim} (точка {point.get('id')})")
            prepared.append(
                models.PointStruct(
                    id=str(point["id"]),
                    vector=vector.tolist(),
                    payload=dict(point["payload"]),
                )
            )
        if not prepared:
            return 0
        step = max(int(batch_size), 1)
        for start in range(0, len(prepared), step):
            self.client.upsert(
                collection_name=self.collection,
                points=prepared[start : start + step],
                wait=True,
            )
        return len(prepared)

    def search(
        self,
        vector: np.ndarray | Sequence[float],
        *,
        limit: int = 10,
        doc_code: str | None = None,
        score_threshold: float | None = None,
    ) -> list[Hit]:
        from qdrant_client import models

        query = np.asarray(vector, dtype=np.float32).ravel()
        query_filter = None
        if doc_code:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="doc_code", match=models.MatchValue(value=doc_code))]
            )
        response = self.client.query_points(
            collection_name=self.collection,
            query=query.tolist(),
            limit=int(limit),
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        return [
            Hit(id=str(point.id), score=float(point.score), payload=dict(point.payload or {}))
            for point in response.points
        ]

    def delete_document(self, doc_code: str) -> int:
        """Удаляет точки документа (переиндексация целиком идёт через upsert)."""
        from qdrant_client import models

        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.Filter(
                    must=[models.FieldCondition(key="doc_code", match=models.MatchValue(value=doc_code))]
                ),
                wait=True,
            )
            return self.count()
        except Exception as exc:  # noqa: BLE001
            log.warning("удаление точек %s не удалось: %s", doc_code, exc)
            return -1

    def count(self) -> int:
        return int(self.client.count(collection_name=self.collection, exact=True).count)

    def available(self) -> bool:
        try:
            return bool(self.client.collection_exists(self.collection))
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None


def point_id_for(doc_code: str, ord_index: int) -> str:
    """Детерминированный id точки: uuid5 от «код документа:порядковый номер»."""
    return str(uuid.uuid5(_NAMESPACE, f"{doc_code}:{ord_index}"))
