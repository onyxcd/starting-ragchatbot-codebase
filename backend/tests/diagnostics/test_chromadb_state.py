"""Diagnostic tests to verify ChromaDB state and data availability"""

import os

import pytest
from config import Config
from vector_store import VectorStore


def test_chromadb_path_exists():
    """Verify that the ChromaDB path exists"""
    config = Config()

    # ChromaDB should create the directory if it doesn't exist,
    # but we can check if it's accessible
    assert config.CHROMA_PATH is not None


def test_chromadb_accessible(vector_store):
    """Verify that ChromaDB can be queried without errors"""
    # Try a simple query - should not crash
    try:
        results = vector_store.course_catalog.get()
        assert results is not None
    except Exception as e:
        pytest.fail(f"ChromaDB query failed: {str(e)}")


def test_chromadb_has_courses():
    """CRITICAL: Verify that courses are loaded in the production database"""
    # This test uses the actual production ChromaDB path
    config = Config()

    # Only run if the production chroma_db exists
    if not os.path.exists(config.CHROMA_PATH):
        pytest.skip(f"Production ChromaDB not found at {config.CHROMA_PATH}")

    store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)
    count = store.get_course_count()

    assert count > 0, (
        f"ChromaDB has no courses loaded. Found {count} courses.\n"
        "Run the application startup event or manually add courses with:\n"
        "rag_system.add_course_folder('../docs')"
    )


def test_chromadb_has_content_chunks():
    """Verify that content chunks exist in the production database"""
    config = Config()

    # Only run if the production chroma_db exists
    if not os.path.exists(config.CHROMA_PATH):
        pytest.skip(f"Production ChromaDB not found at {config.CHROMA_PATH}")

    store = VectorStore(config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS)

    try:
        results = store.course_content.get()
        chunk_count = len(results["ids"]) if results and "ids" in results else 0

        assert chunk_count > 0, (
            f"ChromaDB has no content chunks. Found {chunk_count} chunks.\n"
            "Courses may be in catalog but content was not added."
        )
    except Exception as e:
        pytest.fail(f"Failed to query course_content collection: {str(e)}")


def test_chromadb_collections_exist(vector_store):
    """Verify that both required collections exist"""
    # Check that collections are accessible
    assert vector_store.course_catalog is not None
    assert vector_store.course_content is not None

    # Check that we can query them
    try:
        catalog_results = vector_store.course_catalog.get()
        content_results = vector_store.course_content.get()
        assert catalog_results is not None
        assert content_results is not None
    except Exception as e:
        pytest.fail(f"Failed to access collections: {str(e)}")


def test_vector_store_search_functional(populated_vector_store):
    """Verify that vector store search is functional with test data"""
    results = populated_vector_store.search("machine learning")

    # Should return results or empty, but not error
    assert results is not None
    assert hasattr(results, "documents")
    assert hasattr(results, "metadata")
    assert hasattr(results, "error")

    # With our test data, should find results
    assert not results.is_empty(), "Search returned no results with test data"
    assert results.error is None, f"Search returned error: {results.error}"
