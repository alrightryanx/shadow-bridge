"""
Vector Store Service - Semantic Search for ShadowAI

Uses ChromaDB for local vector storage and sentence-transformers for embeddings.
This enables semantic search where "find my project meeting notes" matches
notes about meetings even if they don't contain those exact words.

AGI-Readiness: This is the foundation for context-aware AI interactions.
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

# Global instances (lazy loaded)
_chroma_client = None
_embedding_model = None
_collection = None

# Configuration
CHROMA_PERSIST_DIR = Path.home() / ".shadowai" / "chroma_db"
COLLECTION_NAME = "shadowai_memory"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Small, fast, good quality

# Feature flag - disable if dependencies aren't installed
VECTOR_STORE_ENABLED = False


def _init_vector_store():
    """Initialize ChromaDB and embedding model lazily."""
    global _chroma_client, _embedding_model, _collection, VECTOR_STORE_ENABLED

    if _chroma_client is not None:
        return True

    try:
        import chromadb
        from chromadb.config import Settings
        from sentence_transformers import SentenceTransformer

        # Create persist directory
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistence
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_PERSIST_DIR),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Get or create collection
        _collection = _chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        # Initialize embedding model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        VECTOR_STORE_ENABLED = True
        logger.info(f"Vector store initialized. Collection has {_collection.count()} items.")
        return True

    except ImportError as e:
        logger.warning(f"Vector store dependencies not installed: {e}")
        logger.warning("Run: pip install chromadb sentence-transformers")
        VECTOR_STORE_ENABLED = False
        return False
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        VECTOR_STORE_ENABLED = False
        return False


def is_available() -> bool:
    """Check if vector store is available."""
    if VECTOR_STORE_ENABLED:
        return True
    return _init_vector_store()


def _generate_id(text: str, source_type: str, source_id: str) -> str:
    """Generate a consistent ID for a document."""
    content = f"{source_type}:{source_id}:{text[:100]}"
    return hashlib.md5(content.encode()).hexdigest()


def embed_text(text: str) -> Optional[List[float]]:
    """
    Generate embedding vector for text.

    Args:
        text: Text to embed

    Returns:
        List of floats representing the embedding, or None if failed
    """
    if not is_available():
        return None

    try:
        embedding = _embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None


def index_document(
    source_type: str,
    source_id: str,
    title: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Index a document for semantic search.

    Args:
        source_type: Type of source (note, project, automation, agent)
        source_id: Unique ID within source type
        title: Document title
        content: Document content (will be chunked if large)
        metadata: Additional metadata to store

    Returns:
        True if indexing succeeded
    """
    if not is_available():
        return False

    try:
        # Combine title and content for embedding
        full_text = f"{title}\n\n{content}"

        # Generate consistent ID
        doc_id = _generate_id(full_text, source_type, source_id)

        # Prepare metadata
        doc_metadata = {
            "source_type": source_type,
            "source_id": source_id,
            "title": title,
            "indexed_at": datetime.now().isoformat(),
            **(metadata or {})
        }

        # Clean metadata - ChromaDB only accepts str, int, float, bool
        clean_metadata = {}
        for k, v in doc_metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_metadata[k] = v
            else:
                clean_metadata[k] = str(v)

        # Upsert to collection (handles both insert and update)
        _collection.upsert(
            ids=[doc_id],
            documents=[full_text],
            metadatas=[clean_metadata]
        )

        logger.debug(f"Indexed {source_type}:{source_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to index document: {e}")
        return False


def remove_document(source_type: str, source_id: str) -> bool:
    """
    Remove a document from the index.

    Args:
        source_type: Type of source
        source_id: ID of document to remove

    Returns:
        True if removal succeeded
    """
    if not is_available():
        return False

    try:
        # Find and delete by metadata
        results = _collection.get(
            where={"$and": [
                {"source_type": source_type},
                {"source_id": source_id}
            ]}
        )

        if results and results.get("ids"):
            _collection.delete(ids=results["ids"])
            logger.debug(f"Removed {source_type}:{source_id}")
            return True
        return False

    except Exception as e:
        logger.error(f"Failed to remove document: {e}")
        return False


def semantic_search(
    query: str,
    source_types: Optional[List[str]] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Perform semantic search across indexed documents.

    Args:
        query: Search query
        source_types: Optional filter by source types
        limit: Maximum results to return

    Returns:
        List of search results with scores and metadata
    """
    if not is_available():
        return []

    try:
        # Build where clause
        where_clause = None
        if source_types:
            if len(source_types) == 1:
                where_clause = {"source_type": source_types[0]}
            else:
                where_clause = {"source_type": {"$in": source_types}}

        # Perform search
        results = _collection.query(
            query_texts=[query],
            n_results=limit,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted = []
        if results and results.get("ids") and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # Convert distance to similarity score (cosine distance -> similarity)
                distance = results["distances"][0][i] if results.get("distances") else 0
                similarity = 1 - distance  # Cosine distance to similarity

                formatted.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "score": round(similarity, 4),
                    "source_type": results["metadatas"][0][i].get("source_type", "unknown") if results.get("metadatas") else "unknown",
                    "source_id": results["metadatas"][0][i].get("source_id", "") if results.get("metadatas") else "",
                    "title": results["metadatas"][0][i].get("title", "") if results.get("metadatas") else ""
                })

        return formatted

    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []


def get_stats() -> Dict[str, Any]:
    """Get vector store statistics."""
    if not is_available():
        return {
            "available": False,
            "error": "Vector store not initialized"
        }

    try:
        count = _collection.count()
        return {
            "available": True,
            "document_count": count,
            "collection_name": COLLECTION_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "persist_directory": str(CHROMA_PERSIST_DIR)
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }


def reindex_all(data_service) -> Dict[str, int]:
    """
    Reindex all content from data service.

    Args:
        data_service: The data_service module with get_* functions

    Returns:
        Dict with counts of indexed items by type
    """
    if not is_available():
        return {"error": "Vector store not available"}

    counts = {"notes": 0, "projects": 0, "automations": 0, "agents": 0}

    try:
        # Index notes
        notes = data_service.get_notes() or []
        for note in notes:
            if index_document(
                source_type="note",
                source_id=note.get("id", ""),
                title=note.get("title", ""),
                content=note.get("content", note.get("preview", "")),
                metadata={"priority": note.get("priority", "normal")}
            ):
                counts["notes"] += 1

        # Index projects
        projects = data_service.get_projects() or []
        for project in projects:
            if index_document(
                source_type="project",
                source_id=project.get("id", project.get("path", "")),
                title=project.get("name", ""),
                content=project.get("description", "") or project.get("path", ""),
                metadata={"path": project.get("path", "")}
            ):
                counts["projects"] += 1

        # Index automations
        automations = data_service.get_automations() or []
        for auto in automations:
            if index_document(
                source_type="automation",
                source_id=auto.get("id", ""),
                title=auto.get("name", ""),
                content=auto.get("description", "") or auto.get("prompt", ""),
                metadata={"trigger": auto.get("trigger_type", "")}
            ):
                counts["automations"] += 1

        # Index agents
        agents = data_service.get_agents() or []
        for agent in agents:
            if index_document(
                source_type="agent",
                source_id=agent.get("id", ""),
                title=agent.get("name", ""),
                content=agent.get("specialty", "") or agent.get("description", ""),
                metadata={"type": agent.get("type", "")}
            ):
                counts["agents"] += 1

        logger.info(f"Reindexed all content: {counts}")
        return counts

    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        return {"error": str(e), **counts}


def clear_index() -> bool:
    """Clear all indexed documents."""
    if not is_available():
        return False

    try:
        # Delete and recreate collection
        _chroma_client.delete_collection(COLLECTION_NAME)
        global _collection
        _collection = _chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Vector index cleared")
        return True
    except Exception as e:
        logger.error(f"Failed to clear index: {e}")
        return False
