"""
rag_memory.py – Módulo de memoria RAG para MIA.

Provee almacenamiento y recuperación de contexto conversacional
usando ChromaDB (vector store local) y sentence-transformers (embeddings).

Responsabilidades:
  - Inicializar / cargar la colección persistente.
  - Ingestar pares (user_msg, assistant_msg) al final de cada turno.
  - Recuperar top-K fragmentos relevantes dado un query.
  - Respetar max_docs eliminando documentos antiguos.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RAGMemory:
    """Memoria conversacional basada en ChromaDB + sentence-transformers."""

    def __init__(self, config: Any) -> None:
        # Accept either a RAGConfig dataclass or a raw dict
        if hasattr(config, "enabled"):
            # Typed RAGConfig dataclass
            self.enabled = config.enabled
            self.top_k = config.top_k
            self.max_docs = config.max_docs
            self.score_threshold = config.score_threshold
            self._persist_dir = config.persist_dir
            self._embedding_model_name = config.embedding_model
        else:
            # Legacy raw dict
            rag_cfg = config.get("rag", {})
            self.enabled = rag_cfg.get("enabled", False)
            self.top_k = rag_cfg.get("top_k", 3)
            self.max_docs = rag_cfg.get("max_docs", 5000)
            self.score_threshold = rag_cfg.get("score_threshold", 0.3)
            self._persist_dir = rag_cfg.get("persist_dir", "./data/chroma_db")
            self._embedding_model_name = rag_cfg.get(
                "embedding_model", "all-MiniLM-L6-v2"
            )

        self._client = None
        self._collection = None
        self._embed_fn = None

        if self.enabled:
            self._initialize()

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------

    def _initialize(self) -> None:
        """Carga ChromaDB y el modelo de embeddings."""
        try:
            import chromadb
            from chromadb.config import Settings

            persist_path = Path(self._persist_dir)
            persist_path.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name="mia_memory",
                metadata={"hnsw:space": "cosine"},
            )

            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._embed_fn = SentenceTransformer(
                self._embedding_model_name, device=device
            )
            logger.info("RAG embeddings device: %s", device)

            logger.info(
                "RAG memory inicializada – docs existentes: %d",
                self._collection.count(),
            )
        except ImportError as exc:
            logger.error("Dependencia RAG faltante: %s", exc)
            self.enabled = False

    # ------------------------------------------------------------------
    # Ingesta
    # ------------------------------------------------------------------

    def ingest(self, user_msg: str, assistant_msg: str) -> None:
        """Almacena un par conversacional como documento."""
        if not self.enabled or self._collection is None:
            return

        doc = f"Usuario: {user_msg}\nMIA: {assistant_msg}"
        doc_id = f"turn_{int(time.time() * 1000)}"

        embedding = self._embed_fn.encode(doc).tolist()
        self._collection.add(
            ids=[doc_id],
            documents=[doc],
            embeddings=[embedding],
            metadatas=[{"timestamp": time.time()}],
        )

        self._enforce_max_docs()
        logger.debug("RAG ingesta completada – id: %s", doc_id)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[str]:
        """Recupera los top-K fragmentos más relevantes para el query."""
        if not self.enabled or self._collection is None:
            return []

        t0 = time.perf_counter()
        embedding = self._embed_fn.encode(query).tolist()

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(self.top_k, self._collection.count() or 1),
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("RAG retrieval: %.1f ms", elapsed_ms)

        docs: list[str] = []
        if results and results.get("documents"):
            for doc_list, dist_list in zip(
                results["documents"], results.get("distances", [[]])
            ):
                for doc, dist in zip(doc_list, dist_list):
                    # ChromaDB cosine distance: 0 = identical, 2 = opposite
                    similarity = 1.0 - (dist / 2.0)
                    if similarity >= self.score_threshold:
                        docs.append(doc)

        return docs

    # ------------------------------------------------------------------
    # Inyección en prompt
    # ------------------------------------------------------------------

    def build_context_block(self, query: str) -> str:
        """Construye el bloque de contexto RAG para inyectar en el prompt."""
        fragments = self.retrieve(query)
        if not fragments:
            return ""

        lines = ["## Contexto previo", ""]
        for i, frag in enumerate(fragments, 1):
            lines.append(f"[{i}] {frag}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Mantenimiento
    # ------------------------------------------------------------------

    def _enforce_max_docs(self) -> None:
        """Elimina documentos antiguos si se excede max_docs."""
        if self._collection is None:
            return

        count = self._collection.count()
        if count <= self.max_docs:
            return

        excess = count - self.max_docs
        all_data = self._collection.get(include=["metadatas"])

        # Ordenar por timestamp y eliminar los más antiguos
        id_ts = sorted(
            zip(all_data["ids"], all_data["metadatas"]),
            key=lambda x: x[1].get("timestamp", 0),
        )

        ids_to_delete = [item[0] for item in id_ts[:excess]]
        self._collection.delete(ids=ids_to_delete)
        logger.info("RAG limpieza: eliminados %d documentos antiguos", excess)

    def clear(self) -> int:
        """Borra toda la memoria vectorizada. Retorna docs eliminados."""
        if not self._client or not self._collection:
            return 0

        count = self._collection.count()
        self._client.delete_collection("mia_memory")
        self._collection = self._client.get_or_create_collection(
            name="mia_memory",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("RAG: %d documentos eliminados", count)
        return count

    def get_3d_points(self) -> list[dict]:
        """Retorna todos los docs con coordenadas 3D (PCA de embeddings)."""
        if not self._client or not self._collection:
            return []

        count = self._collection.count()
        if count == 0:
            return []

        # Obtener todos los datos
        data = self._collection.get(
            include=["embeddings", "documents", "metadatas"],
        )

        embeddings = data.get("embeddings", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])

        if not embeddings or len(embeddings) < 2:
            # Con < 2 puntos PCA no tiene sentido
            points = []
            for i, doc in enumerate(documents):
                ts = metadatas[i].get("timestamp", 0) if metadatas else 0
                points.append({
                    "x": 0, "y": 0, "z": 0,
                    "text": doc[:200] if doc else "",
                    "timestamp": ts,
                })
            return points

        try:
            from sklearn.decomposition import PCA
            import numpy as np

            emb_array = np.array(embeddings)
            n_components = min(3, emb_array.shape[0], emb_array.shape[1])
            pca = PCA(n_components=n_components)
            coords_3d = pca.fit_transform(emb_array)

            # Normalizar a rango [-1, 1] para la escena
            if coords_3d.max() != coords_3d.min():
                coords_3d = (coords_3d - coords_3d.mean(axis=0)) / (coords_3d.std(axis=0) + 1e-8)

            points = []
            for i in range(len(documents)):
                ts = metadatas[i].get("timestamp", 0) if metadatas else 0
                x = float(coords_3d[i, 0]) if n_components > 0 else 0
                y = float(coords_3d[i, 1]) if n_components > 1 else 0
                z = float(coords_3d[i, 2]) if n_components > 2 else 0
                points.append({
                    "x": x, "y": y, "z": z,
                    "text": documents[i][:200] if documents[i] else "",
                    "timestamp": ts,
                })
            return points

        except ImportError:
            logger.warning("sklearn no disponible para PCA")
            return []
