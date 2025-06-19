import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available, falling back to simple vector store")

from .llm_client import llm_client

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a semantically meaningful chunk of a document"""

    def __init__(
        self,
        text: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None,
        chunk_id: Optional[str] = None,
    ):
        self.text = text
        self.metadata = metadata
        self.embedding = embedding
        # Generate unique ID based on content if not provided
        self.chunk_id = chunk_id or hashlib.md5(text.encode()).hexdigest()
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()  # Track last access time
        # Calculate text hash for deduplication
        self.text_hash = self._semantic_hash(text)

    def _semantic_hash(self, text: str) -> str:
        """Create a robust hash of normalized document content for deduplication"""
        # Normalize text: lowercase, remove punctuation, normalize whitespace
        import re
        import string

        # Remove all punctuation except apostrophes (for contractions)
        translator = str.maketrans("", "", string.punctuation.replace("'", ""))
        normalized = text.lower().translate(translator)

        # Normalize whitespace and remove extra spaces
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # Hash the entire normalized content
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def is_similar_to(self, other: "DocumentChunk", threshold: float = 0.8) -> bool:
        """Check if this chunk is semantically similar to another"""
        # Quick check based on semantic hash
        if self.text_hash == other.text_hash:
            return True

        # If embeddings available, compare semantic similarity
        if self.embedding and other.embedding:
            similarity = self._cosine_similarity(self.embedding, other.embedding)
            return similarity > threshold

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "text_hash": self.text_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        """Create from dictionary representation"""
        chunk = cls(
            text=data["text"],
            metadata=data["metadata"],
            chunk_id=data["chunk_id"],
        )
        chunk.text_hash = data["text_hash"]
        chunk.created_at = datetime.fromisoformat(data["created_at"])
        return chunk


class SmartChunker:
    """Intelligent document chunking with semantic boundaries"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ":",
            " - ",
            "\t",
            "  ",
        ]

    def chunk_document(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Split document into semantically meaningful chunks"""
        chunks: List[DocumentChunk] = []  # Already properly typed

        # Handle empty or very short documents
        if not text or len(text) < self.chunk_size / 2:
            return [DocumentChunk(text=text, metadata=metadata.copy())]

        # Split text using semantic boundaries
        current_pos = 0
        while current_pos < len(text):
            # Find end position for this chunk
            end_pos = min(current_pos + self.chunk_size, len(text))

            # Try to find a natural boundary near the end position
            if end_pos < len(text):
                for separator in self.separators:
                    boundary = text.rfind(separator, current_pos, end_pos)
                    if boundary != -1:
                        end_pos = boundary + len(separator)
                        break

            # Extract chunk text
            chunk_text = text[current_pos:end_pos].strip()
            if chunk_text:
                # Create chunk with metadata including position info
                chunk_metadata = metadata.copy()
                chunk_metadata.update(
                    {
                        "start_pos": current_pos,
                        "end_pos": end_pos,
                        "chunk_index": len(chunks),
                    }
                )
                chunks.append(DocumentChunk(text=chunk_text, metadata=chunk_metadata))

            # Move position for next chunk, with overlap
            next_pos = end_pos - self.chunk_overlap
            current_pos = max(current_pos + 1, next_pos)

        return chunks


class EnhancedVectorStore:
    """Optimized vector store with smart chunking, deduplication and incremental indexing"""

    def __init__(
        self,
        embedding_dim: int = 384,
        index_path: Optional[str] = None,
        persist_dir: Optional[str] = None,
    ):
        self.embedding_dim = embedding_dim
        self.chunks: List[DocumentChunk] = []
        self.chunker = SmartChunker()
        self.index_path = index_path
        self.persist_dir = persist_dir
        self.last_save_time = datetime.now()
        self.chunks_since_save = 0
        self.chunk_ids_to_index: Dict[str, int] = (
            {}
        )  # Maps chunk IDs to index positions
        self._lock = asyncio.Lock()  # Async lock for thread safety
        self.max_chunks = 10000  # Maximum chunks before cleanup
        self.cleanup_threshold = 0.8  # Cleanup when reaching 80% of max_chunks

        # Initialize FAISS index if available
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.use_faiss = True
        else:
            self.use_faiss = False

        # Load existing index if path provided
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between vectors"""
        dot_product = sum(x * y for x, y in zip(v1, v2))
        norm1 = sum(x * x for x in v1) ** 0.5
        norm2 = sum(x * x for x in v2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 * norm2 > 0 else 0

    async def add_document(
        self, text: str, metadata: Dict[str, Any], auto_embed: bool = True
    ) -> List[str]:
        """Add a document, automatically chunking and embedding it"""
        # Generate chunks
        chunks = self.chunker.chunk_document(text, metadata)
        chunk_ids = []

        # Process each chunk
        for chunk in chunks:
            # Check for duplication
            is_duplicate = False
            for existing_chunk in self.chunks:
                if chunk.is_similar_to(existing_chunk):
                    logger.debug(f"Skipping duplicate chunk: {chunk.chunk_id[:8]}")
                    is_duplicate = True
                    break

            if not is_duplicate:
                # Generate embedding if needed (outside lock since it's async)
                if auto_embed and not chunk.embedding:
                    chunk.embedding = await llm_client.embed(
                        model="gemma3:12b", text=chunk.text
                    )

                # Protect critical section with lock
                async with self._lock:
                    # Add to store
                    self.chunks.append(chunk)
                    self.chunk_ids_to_index[chunk.chunk_id] = len(self.chunks) - 1
                    chunk_ids.append(chunk.chunk_id)

                    # Add to FAISS index if available
                    if self.use_faiss and chunk.embedding:
                        embedding_np = np.array([chunk.embedding], dtype=np.float32)
                        faiss.normalize_L2(embedding_np)
                        self.index.add(embedding_np)

                    self.chunks_since_save += 1

        # Auto-save if many changes accumulated
        if self.persist_dir and self.chunks_since_save > 50:
            await self.save_index_async()

        return chunk_ids

    async def add_documents_async(
        self, documents: List[Tuple[str, Dict[str, Any]]], batch_size: int = 5
    ) -> List[str]:
        """Add multiple documents asynchronously with batching"""
        all_chunk_ids = []

        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            tasks = []

            for text, metadata in batch:
                tasks.append(self.add_document(text, metadata))

            # Execute batch concurrently
            results = await asyncio.gather(*tasks)

            # Collect all chunk IDs
            for chunk_ids in results:
                all_chunk_ids.extend(chunk_ids)

        return all_chunk_ids

    async def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.65,
    ) -> List[Dict[str, Any]]:
        """Search for relevant document chunks"""
        # Get query embedding
        query_embedding = await llm_client.embed(model="gemma3:12b", text=query)

        if self.use_faiss and len(self.chunks) > 0:
            # Use FAISS for fast retrieval (get more results for filtering)
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)

            # Protect FAISS search with lock
            async with self._lock:
                D, result_indices = self.index.search(
                    query_vector, min(k * 3, len(self.chunks))
                )

            # Get candidates
            candidates = []
            for idx, (distance_idx, result_index) in enumerate(
                zip(D[0], result_indices[0])
            ):
                if result_index < len(self.chunks):
                    chunk = self.chunks[result_index]
                    # Apply metadata filter if specified
                    if filter_metadata and not self._matches_filter(
                        chunk.metadata, filter_metadata
                    ):
                        continue
                    similarity = (
                        1.0 - distance_idx / 2
                    )  # Convert L2 distance to similarity
                    candidates.append((chunk, similarity))

            # Filter by similarity threshold
            candidates = [
                (chunk, similarity)
                for chunk, similarity in candidates
                if similarity >= min_similarity
            ]
        else:
            # Fallback to manual search
            candidates = []
            for chunk in self.chunks:
                # Skip if doesn't match filter
                if filter_metadata and not self._matches_filter(
                    chunk.metadata, filter_metadata
                ):
                    continue

                # Skip if no embedding
                if not chunk.embedding:
                    continue

                # Calculate similarity
                similarity = self._cosine_similarity(query_embedding, chunk.embedding)
                if similarity >= min_similarity:
                    candidates.append((chunk, similarity))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Format results
        results = []
        for chunk, similarity in candidates[:k]:
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "similarity": similarity,
                }
            )

        return results

    def _matches_filter(
        self, metadata: Dict[str, Any], filter_criteria: Dict[str, Any]
    ) -> bool:
        """Check if metadata matches filter criteria"""
        for key, val in filter_criteria.items():
            if key not in metadata:
                return False

            if isinstance(val, list):
                # List means "any of these values"
                if metadata[key] not in val:
                    return False
            elif metadata[key] != val:
                return False

        return True

    async def save_index_async(self) -> None:
        """Save index and chunks asynchronously"""
        if not self.persist_dir:
            return

        # Check if we need to cleanup before saving
        if len(self.chunks) > self.max_chunks * self.cleanup_threshold:
            removed = await self.cleanup_old_chunks()
            logger.info(f"Pre-save cleanup removed {removed} old chunks")

        # Create directory if it doesn't exist
        os.makedirs(self.persist_dir, exist_ok=True)

        # Prepare chunk data for saving
        chunks_data = [chunk.to_dict() for chunk in self.chunks]

        # Define path for chunks data
        chunks_path = os.path.join(self.persist_dir, "chunks.json")

        # Use asyncio to run file operations without blocking
        def _save_data():
            # Save chunks data
            with open(chunks_path, "w") as f:
                json.dump(chunks_data, f)

            # Save FAISS index if available
            if self.use_faiss:
                index_path = os.path.join(self.persist_dir, "faiss.index")
                faiss.write_index(self.index, index_path)

        await asyncio.to_thread(_save_data)

        self.last_save_time = datetime.now()
        self.chunks_since_save = 0
        logger.info(f"Vector index saved to {self.persist_dir}")

    @lru_cache(maxsize=1000)
    def get_chunk(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Get a chunk by ID with LRU caching"""
        if chunk_id not in self.chunk_ids_to_index:
            return None
        chunk = self.chunks[self.chunk_ids_to_index[chunk_id]]
        chunk.last_accessed = datetime.now()  # Update access time
        return chunk

    async def cleanup_old_chunks(self, max_age_days: int = 30) -> int:
        """Remove chunks not accessed in specified days"""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        removed_count = 0

        async with self._lock:
            # Filter chunks to keep
            new_chunks = []
            new_mapping = {}
            for i, chunk in enumerate(self.chunks):
                if chunk.last_accessed >= cutoff:
                    new_mapping[chunk.chunk_id] = len(new_chunks)
                    new_chunks.append(chunk)
                else:
                    removed_count += 1

            # Update store
            self.chunks = new_chunks
            self.chunk_ids_to_index = new_mapping

            # Clear cache to remove deleted chunks
            self.get_chunk.cache_clear()

        logger.info(f"Cleaned up {removed_count} old chunks")
        return removed_count

    def load_index(self, directory: str) -> None:
        """Load index and chunks from directory"""
        chunks_path = os.path.join(directory, "chunks.json")

        # Load chunks data
        if os.path.exists(chunks_path):
            try:
                with open(chunks_path, "r") as f:
                    chunks_data = json.load(f)

                self.chunks = [DocumentChunk.from_dict(data) for data in chunks_data]
                self.chunk_ids_to_index = {
                    chunk.chunk_id: i for i, chunk in enumerate(self.chunks)
                }
                logger.info(f"Loaded {len(self.chunks)} chunks from {chunks_path}")
            except Exception as e:
                logger.error(f"Error loading chunks data: {e}")

        # Load FAISS index if available
        if self.use_faiss:
            index_path = os.path.join(directory, "faiss.index")
            if os.path.exists(index_path):
                try:
                    self.index = faiss.read_index(index_path)
                    logger.info(f"Loaded FAISS index from {index_path}")
                except Exception as e:
                    logger.error(f"Error loading FAISS index: {e}")


class AsyncDocumentLoader:
    """Asynchronous document loader with incremental indexing"""

    def __init__(self, vector_store: EnhancedVectorStore):
        self.vector_store = vector_store
        self.loading_task = None
        self._stop_event = asyncio.Event()
        self.last_modified_times: Dict[str, float] = {}

    async def load_directory(
        self,
        directory: str,
        glob_pattern: str = "**/*.*",
        metadata_fn: Optional[Callable] = None,
    ) -> None:
        """Load all documents from a directory with pattern matching"""
        # Collect all matching files
        path = Path(directory)
        files = list(path.glob(glob_pattern))

        loaded_count = 0
        for file_path in files:
            if self._stop_event.is_set():
                logger.info("Loading interrupted")
                break

            try:
                # Skip non-text files
                if file_path.suffix.lower() not in [
                    ".txt",
                    ".md",
                    ".csv",
                    ".json",
                    ".html",
                    ".xml",
                    ".py",
                    ".js",
                ]:
                    continue

                # Read file content
                text = file_path.read_text(errors="replace")

                # Generate metadata
                if metadata_fn:
                    metadata = metadata_fn(file_path)
                else:
                    metadata = {
                        "source": str(file_path),
                        "filename": file_path.name,
                        "extension": file_path.suffix,
                        "last_modified": datetime.fromtimestamp(
                            file_path.stat().st_mtime
                        ).isoformat(),
                    }

                # Add to vector store
                await self.vector_store.add_document(text, metadata)
                loaded_count += 1

                # Periodically log progress
                if loaded_count % 10 == 0:
                    logger.info(f"Loaded {loaded_count}/{len(files)} documents")

            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        # Save index after loading all documents
        await self.vector_store.save_index_async()
        logger.info(f"Completed loading {loaded_count} documents")

    async def start_incremental_indexing(
        self,
        directory: str,
        glob_pattern: str = "**/*.*",
        check_interval: int = 300,  # seconds
        metadata_fn: Optional[Callable] = None,
    ) -> None:
        """Start background task for incremental indexing of new/modified files"""
        self._stop_event.clear()

        # Track last modified times
        self.last_modified_times: Dict[str, float] = {}

        async def _indexing_task():
            while not self._stop_event.is_set():
                try:
                    # Collect all matching files
                    path = Path(directory)
                    files = list(path.glob(glob_pattern))

                    for file_path in files:
                        # Skip non-text files
                        if file_path.suffix.lower() not in [
                            ".txt",
                            ".md",
                            ".csv",
                            ".json",
                            ".html",
                            ".xml",
                            ".py",
                            ".js",
                        ]:
                            continue

                        file_str = str(file_path)
                        mtime = file_path.stat().st_mtime

                        # Check if file is new or modified
                        if (
                            file_str not in self.last_modified_times
                            or self.last_modified_times[file_str] < mtime
                        ):
                            try:
                                # Read file content
                                text = file_path.read_text(errors="replace")

                                # Generate metadata
                                if metadata_fn:
                                    metadata = metadata_fn(file_path)
                                else:
                                    metadata = {
                                        "source": file_str,
                                        "filename": file_path.name,
                                        "extension": file_path.suffix,
                                        "last_modified": datetime.fromtimestamp(
                                            mtime
                                        ).isoformat(),
                                    }

                                # Add to vector store
                                await self.vector_store.add_document(text, metadata)
                                self.last_modified_times[file_str] = mtime
                                logger.info(f"Indexed new/modified file: {file_path}")

                            except Exception as e:
                                logger.error(f"Error indexing {file_path}: {e}")

                    # Check for any deleted files and remove from tracking
                    current_files = set(str(f) for f in files)
                    tracked_files = set(self.last_modified_times.keys())
                    for deleted_file in tracked_files - current_files:
                        del self.last_modified_times[deleted_file]

                    # Save index periodically
                    if self.vector_store.chunks_since_save > 0:
                        await self.vector_store.save_index_async()

                    # Wait for next check interval
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(), timeout=check_interval
                        )
                    except asyncio.TimeoutError:
                        pass  # This is expected when timeout occurs

                except Exception as e:
                    logger.error(f"Error in incremental indexing: {e}")
                    # Wait before retrying
                    await asyncio.sleep(60)


# Create a global instance of EnhancedVectorStore
enhanced_vector_store = EnhancedVectorStore()
