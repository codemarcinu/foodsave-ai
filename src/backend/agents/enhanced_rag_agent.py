"""
Enhanced RAG Agent

This agent implements advanced Retrieval-Augmented Generation capabilities:
- Supports multiple document types and sources
- Uses enhanced vector storage and retrieval
- Handles document chunking and embedding
- Provides source tracking and attribution
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..agents.base_agent import AgentResponse, BaseAgent
from ..core.enhanced_vector_store import enhanced_vector_store
from ..core.hybrid_llm_client import ModelComplexity, hybrid_llm_client
from ..core.rag_document_processor import rag_document_processor

logger = logging.getLogger(__name__)


class EnhancedRAGAgent(BaseAgent):
    """
    Enhanced Retrieval-Augmented Generation Agent

    Features:
    - Document processing with smart chunking
    - Efficient vector storage and retrieval
    - Support for various document formats
    - Source tracking and attribution
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.initialized = False
        self.document_processor = rag_document_processor
        self.vector_store = enhanced_vector_store

    async def initialize(self):
        """Initialize the agent by ensuring vector store is populated"""
        if not self.initialized:
            # Check if vector store has documents
            if not self.vector_store.chunks:
                # Try to process documents in the data/docs directory
                docs_dir = Path("data/docs")
                if docs_dir.exists() and docs_dir.is_dir():
                    logger.info(
                        f"Initializing RAG agent with documents from {docs_dir}"
                    )
                    await self.document_processor.process_directory(
                        docs_dir,
                        file_extensions=[".txt", ".pdf", ".docx", ".md", ".html"],
                        recursive=True,
                    )

            self.initialized = True
            logger.info(
                f"RAG agent initialized with {len(self.vector_store.chunks)} chunks"
            )

    async def add_document(
        self, content: str, source_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a document to the knowledge base

        Args:
            content: Document text content
            source_id: Source identifier
            metadata: Optional metadata about the document

        Returns:
            Processing result information
        """
        return await self.document_processor.process_document(
            content, source_id, metadata
        )

    async def add_file(
        self, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a document file to the knowledge base

        Args:
            file_path: Path to the document file
            metadata: Optional metadata about the document

        Returns:
            Processing result information
        """
        return await self.document_processor.process_file(file_path, metadata)

    async def add_url(
        self, url: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add content from a URL to the knowledge base

        Args:
            url: URL to fetch content from
            metadata: Optional metadata about the content

        Returns:
            Processing result information
        """
        return await self.document_processor.process_url(url, metadata)

    async def add_directory(
        self,
        directory_path: Union[str, Path],
        file_extensions: List[str] = None,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """
        Add all documents in a directory to the knowledge base

        Args:
            directory_path: Path to the directory
            file_extensions: List of file extensions to include
            recursive: Whether to process subdirectories

        Returns:
            Processing result information
        """
        if file_extensions is None:
            file_extensions = [".txt", ".pdf", ".docx", ".md", ".html"]

        return await self.document_processor.process_directory(
            directory_path, file_extensions=file_extensions, recursive=recursive
        )

    async def search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.65,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks

        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filter
            min_similarity: Minimum similarity threshold

        Returns:
            List of relevant document chunks with metadata
        """
        await self.initialize()
        return await self.vector_store.search(
            query=query,
            k=k,
            filter_metadata=filter_metadata,
            min_similarity=min_similarity,
        )

    async def process(self, context: Dict[str, Any]) -> AgentResponse:
        """
        Process a query using RAG

        Args:
            context: Request context with query

        Returns:
            AgentResponse with answer
        """
        await self.initialize()

        query = context.get("query", "")
        if not query:
            return AgentResponse(text="No query provided.", data={}, success=False)

        # Get query embedding and search for relevant chunks
        retrieved_docs = await self.search(query, k=5)

        if not retrieved_docs:
            return AgentResponse(
                text="I don't have enough information to answer that question.",
                data={},
                success=False,
            )

        # Format context from retrieved documents
        context_chunks = []
        sources = []

        for i, doc in enumerate(retrieved_docs):
            chunk_text = doc["text"]
            source = doc["metadata"].get("source", "unknown")
            # similarity = doc["similarity"]  # Unused variable

            # Add to context
            context_chunks.append(f"[Chunk {i+1}] {chunk_text}")

            # Track source if not already included
            if source not in sources:
                sources.append(source)

        context_text = "\n\n".join(context_chunks)

        # Create prompt with context
        prompt = f"""Answer the question based on the following context information, and if the answer is not contained within the context, say "I don't have enough information to answer that question."

Context:
{context_text}

Question: {query}

Answer:"""

        # Get response from LLM
        response = await hybrid_llm_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on the provided context. Include relevant details from the context, but don't add information that isn't there.",
                },
                {"role": "user", "content": prompt},
            ],
            force_complexity=ModelComplexity.COMPLEX,
            stream=False,
        )

        # Format the response
        if response and "message" in response:
            answer = response["message"]["content"]

            # Add source attribution
            sources_text = ""
            if sources:
                sources_text = "\n\nSources:\n" + "\n".join(f"- {s}" for s in sources)

            full_answer = f"{answer}{sources_text}"

            return AgentResponse(
                text=full_answer,
                data={"retrieved_docs": retrieved_docs, "sources": sources},
                success=True,
            )
        else:
            return AgentResponse(
                text="Failed to generate an answer from the context.",
                data={},
                success=False,
            )
