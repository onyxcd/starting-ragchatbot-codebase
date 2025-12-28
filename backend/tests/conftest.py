"""Shared test fixtures for the RAG chatbot test suite"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Generator
from unittest.mock import Mock, MagicMock
import anthropic
import sys
import os

# Add backend directory to path so we can import modules
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from ai_generator import AIGenerator
from session_manager import SessionManager
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk
from config import Config


@pytest.fixture(scope="function")
def temp_chroma_dir() -> Generator[str, None, None]:
    """Create temporary ChromaDB directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def test_config(temp_chroma_dir: str) -> Config:
    """Test configuration with temp paths and test API key"""
    config = Config()
    config.CHROMA_PATH = temp_chroma_dir
    config.ANTHROPIC_API_KEY = "test-api-key-12345"
    config.MAX_RESULTS = 3  # Smaller for faster tests
    return config


@pytest.fixture(scope="function")
def vector_store(temp_chroma_dir: str) -> VectorStore:
    """Clean VectorStore instance for testing"""
    return VectorStore(
        chroma_path=temp_chroma_dir,
        embedding_model="all-MiniLM-L6-v2",
        max_results=3
    )


@pytest.fixture(scope="function")
def sample_course() -> Course:
    """Sample course for testing"""
    return Course(
        title="Introduction to Machine Learning",
        instructor="Dr. Smith",
        course_link="https://example.com/ml-course",
        lessons=[
            Lesson(lesson_number=1, title="What is ML?", lesson_link="https://example.com/ml-lesson1"),
            Lesson(lesson_number=2, title="Supervised Learning", lesson_link="https://example.com/ml-lesson2"),
            Lesson(lesson_number=3, title="Neural Networks", lesson_link="https://example.com/ml-lesson3"),
        ]
    )


@pytest.fixture(scope="function")
def sample_chunks(sample_course: Course) -> list[CourseChunk]:
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence focused on data-driven algorithms.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Supervised learning uses labeled training data to learn patterns and make predictions.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Neural networks are computational models inspired by biological neural networks.",
            course_title=sample_course.title,
            lesson_number=3,
            chunk_index=2
        ),
    ]


@pytest.fixture(scope="function")
def populated_vector_store(vector_store: VectorStore, sample_course: Course, sample_chunks: list[CourseChunk]) -> VectorStore:
    """VectorStore with sample data loaded"""
    vector_store.add_course_metadata(sample_course)
    vector_store.add_course_content(sample_chunks)
    return vector_store


@pytest.fixture(scope="function")
def course_search_tool(populated_vector_store: VectorStore) -> CourseSearchTool:
    """CourseSearchTool with populated data"""
    return CourseSearchTool(populated_vector_store)


@pytest.fixture(scope="function")
def course_outline_tool(populated_vector_store: VectorStore) -> CourseOutlineTool:
    """CourseOutlineTool with populated data"""
    return CourseOutlineTool(populated_vector_store)


@pytest.fixture(scope="function")
def tool_manager(course_search_tool: CourseSearchTool, course_outline_tool: CourseOutlineTool) -> ToolManager:
    """ToolManager with both tools registered"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    manager.register_tool(course_outline_tool)
    return manager


@pytest.fixture(scope="function")
def mock_anthropic_client(monkeypatch):
    """Mock Anthropic API client to avoid real API calls"""
    mock_client = MagicMock(spec=anthropic.Anthropic)

    # Create a mock response object for non-tool responses
    mock_response = MagicMock()
    mock_text_block = MagicMock()
    mock_text_block.text = "Test response"
    mock_text_block.type = "text"
    mock_response.content = [mock_text_block]
    mock_response.stop_reason = "end_turn"

    mock_client.messages.create.return_value = mock_response

    # Patch the Anthropic constructor
    monkeypatch.setattr("anthropic.Anthropic", lambda api_key: mock_client)

    return mock_client


@pytest.fixture(scope="function")
def ai_generator(test_config: Config, mock_anthropic_client) -> AIGenerator:
    """AIGenerator with mocked Anthropic client"""
    return AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)


@pytest.fixture(scope="function")
def rag_system(test_config: Config, mock_anthropic_client) -> RAGSystem:
    """Full RAGSystem with mocked API"""
    return RAGSystem(test_config)


@pytest.fixture(scope="function")
def session_manager() -> SessionManager:
    """SessionManager instance"""
    return SessionManager(max_history=2)


@pytest.fixture(scope="function")
def test_app(test_config: Config, mock_anthropic_client):
    """FastAPI test app without static file mounting to avoid file existence issues"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Dict
    import anthropic

    # Initialize test FastAPI app
    app = FastAPI(title="Course Materials RAG System (Test)", root_path="")

    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Initialize test RAG system
    test_rag_system = RAGSystem(test_config)

    # Define request/response models (same as in app.py)
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Dict[str, Optional[str]]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # Define API endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = test_rag_system.session_manager.create_session()

            answer, sources = test_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except anthropic.AuthenticationError as e:
            raise HTTPException(status_code=500, detail=f"Authentication error: {str(e)}")
        except anthropic.RateLimitError as e:
            raise HTTPException(status_code=500, detail=f"Rate limit exceeded: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = test_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"message": "Test API is running"}

    # Attach rag_system for test patching
    app.state.rag_system = test_rag_system

    return app


@pytest.fixture(scope="function")
def test_client(test_app):
    """FastAPI test client using the test app"""
    from fastapi.testclient import TestClient
    return TestClient(test_app)
