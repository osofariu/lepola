"""
Embedding and FAISS management for document ingestion.

- Paragraph chunking
- Embedding via multiple providers (Ollama, OpenAI, AWS Bedrock, etc.)
- FAISS index management (save/load)
- Async support for embedding/indexing
"""

import os
import logging
from typing import List, Tuple, Optional
from uuid import uuid4
from datetime import datetime
import faiss
import numpy as np

from src.core.models import Embedding
from src.core.repository import embedding_repository
from src.ai.embeddings_service import embedding_service, EmbeddingServiceError

FAISS_INDEX_DIR = "data/faiss_indexes"
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_DIR, "main.index")

logger = logging.getLogger(__name__)


def chunk_paragraphs(text: str) -> List[Tuple[str, int, int]]:
    """Chunk text into paragraphs.

    Args:
        text: The full document text.

    Returns:
        List of tuples: (paragraph_text, start_pos, end_pos)
    """
    paragraphs = []
    start = 0
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            start += 2  # skip double newline
            continue
        end = start + len(para)
        paragraphs.append((para, start, end))
        start = end + 2  # account for double newline
    return paragraphs


async def embed_texts(paragraphs: List[str]) -> List[List[float]]:
    """Get embeddings for paragraphs using the embedding service.

    Args:
        paragraphs: List of paragraph texts.

    Returns:
        List of embedding vectors (lists of floats).
    """
    try:
        return await embedding_service.embed_texts(paragraphs)
    except EmbeddingServiceError as e:
        logger.error(f"Embedding service failed: {e}")
        raise


def save_faiss_index(index: faiss.Index, path: str = FAISS_INDEX_FILE) -> None:
    """Save FAISS index to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)
    logger.info(f"FAISS index saved to {path}")


def load_faiss_index(path: str = FAISS_INDEX_FILE) -> Optional[faiss.Index]:
    """Load FAISS index from disk, or return None if not found."""
    if not os.path.exists(path):
        return None
    return faiss.read_index(path)


def create_or_update_faiss_index(
    embeddings: List[List[float]], index_path: str = FAISS_INDEX_FILE
) -> faiss.Index:
    """Create or update a FAISS index with new embeddings."""
    arr = np.array(embeddings).astype("float32")
    index = load_faiss_index(index_path)
    if index is None:
        index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    save_faiss_index(index, index_path)
    return index


async def process_document_embeddings(document_id: str, text: str) -> List[Embedding]:
    """Chunk, embed, and index a document asynchronously.

    Args:
        document_id: The document's ID.
        text: The full document text.

    Returns:
        List of Embedding records created.
    """
    # 1. Chunk
    chunks = chunk_paragraphs(text)
    chunk_texts = [c[0] for c in chunks]

    # 2. Embed using the embedding service
    vectors = await embed_texts(chunk_texts)

    # 3. Index
    index = create_or_update_faiss_index(vectors)

    # 4. Store mapping
    embeddings = []
    for i, (chunk, (text, start, end)) in enumerate(zip(vectors, chunks)):
        emb = Embedding(
            document_id=document_id,
            chunk_id=str(uuid4()),
            vector_id=i,  # index in FAISS
            chunk_text=text,
            start_pos=start,
            end_pos=end,
            created_at=datetime.utcnow(),
        )
        embedding_repository.create(emb)
        embeddings.append(emb)
    return embeddings
