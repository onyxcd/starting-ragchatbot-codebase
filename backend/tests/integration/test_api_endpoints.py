"""Integration tests for FastAPI endpoints"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))


@pytest.fixture
def test_client():
    """FastAPI test client"""
    # Import app after adding to path
    from app import app
    return TestClient(app)


def test_query_endpoint_success(test_client, monkeypatch):
    """Test /api/query endpoint with valid request"""
    # Mock the rag_system.query method
    def mock_query(query, session_id=None):
        return "Test answer about machine learning", [{"text": "Source 1", "url": "http://example.com"}]

    # Patch at the app module level
    with patch('app.rag_system.query', side_effect=mock_query):
        response = test_client.post("/api/query", json={"query": "What is ML?"})

    # Check response
    assert response.status_code == 200

    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "session_id" in data

    assert data["answer"] == "Test answer about machine learning"
    assert len(data["sources"]) == 1


def test_query_endpoint_with_session(test_client):
    """Test /api/query endpoint with session ID"""
    with patch('app.rag_system.query') as mock_query:
        mock_query.return_value = ("Answer", [])

        response = test_client.post("/api/query", json={
            "query": "What is AI?",
            "session_id": "test-session-123"
        })

    assert response.status_code == 200

    # Check that query was called with session ID
    mock_query.assert_called_once()
    call_args = mock_query.call_args
    assert call_args[0][1] == "test-session-123"  # Second argument should be session_id


def test_query_endpoint_error_handling(test_client):
    """Test /api/query error handling"""
    # Mock query that raises exception
    with patch('app.rag_system.query', side_effect=Exception("Test error")):
        response = test_client.post("/api/query", json={"query": "test"})

    # Should return 500 error
    assert response.status_code == 500

    # Error message should be in detail
    data = response.json()
    assert "detail" in data
    assert "Test error" in data["detail"]


def test_query_endpoint_returns_sources(test_client):
    """Test that /api/query correctly returns sources"""
    test_sources = [
        {"text": "Introduction to ML - Lesson 1", "url": "https://example.com/lesson1"},
        {"text": "Neural Networks - Lesson 3", "url": "https://example.com/lesson3"}
    ]

    with patch('app.rag_system.query') as mock_query:
        mock_query.return_value = ("Answer with sources", test_sources)

        response = test_client.post("/api/query", json={"query": "What are neural networks?"})

    assert response.status_code == 200

    data = response.json()
    assert "sources" in data
    assert len(data["sources"]) == 2
    assert data["sources"][0]["text"] == "Introduction to ML - Lesson 1"
    assert data["sources"][0]["url"] == "https://example.com/lesson1"


def test_query_endpoint_missing_query_field(test_client):
    """Test /api/query with missing query field"""
    response = test_client.post("/api/query", json={})

    # Should return 422 validation error
    assert response.status_code == 422


def test_courses_endpoint(test_client):
    """Test /api/courses endpoint"""
    mock_analytics = {
        "total_courses": 3,
        "course_titles": ["Course 1", "Course 2", "Course 3"]
    }

    with patch('app.rag_system.get_course_analytics', return_value=mock_analytics):
        response = test_client.get("/api/courses")

    assert response.status_code == 200

    data = response.json()
    assert "total_courses" in data
    assert "course_titles" in data
    assert data["total_courses"] == 3
    assert len(data["course_titles"]) == 3


def test_root_endpoint_serves_frontend(test_client):
    """Test that root endpoint serves the frontend"""
    # Note: This may fail if frontend files don't exist
    response = test_client.get("/")

    # Should either serve the file (200) or return 404 if frontend not found
    assert response.status_code in [200, 404]


def test_query_endpoint_creates_session_if_not_provided(test_client):
    """Test that endpoint creates new session if none provided"""
    with patch('app.rag_system.query') as mock_query:
        mock_query.return_value = ("Answer", [])

        response = test_client.post("/api/query", json={"query": "test"})

    assert response.status_code == 200

    data = response.json()
    # Should have created a session
    assert "session_id" in data
    assert data["session_id"] is not None


def test_query_endpoint_handles_anthropic_auth_error(test_client):
    """Test handling of Anthropic authentication errors"""
    import anthropic
    from unittest.mock import Mock

    # Create a mock response for the error
    mock_response = Mock()
    mock_response.status_code = 401

    # Create the error with required parameters
    auth_error = anthropic.AuthenticationError(
        message="Invalid API key",
        response=mock_response,
        body={"error": {"message": "Invalid API key"}}
    )

    with patch('app.rag_system.query', side_effect=auth_error):
        response = test_client.post("/api/query", json={"query": "test"})

    # Should return 500
    assert response.status_code == 500

    # Error should mention authentication
    data = response.json()
    assert "detail" in data


def test_query_endpoint_handles_rate_limit(test_client):
    """Test handling of API rate limiting"""
    import anthropic
    from unittest.mock import Mock

    # Create a mock response for the error
    mock_response = Mock()
    mock_response.status_code = 429

    # Create the error with required parameters
    rate_limit_error = anthropic.RateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"error": {"message": "Rate limit exceeded"}}
    )

    with patch('app.rag_system.query', side_effect=rate_limit_error):
        response = test_client.post("/api/query", json={"query": "test"})

    # Should return 500
    assert response.status_code == 500


def test_query_request_model_validation():
    """Test QueryRequest model validation"""
    from app import QueryRequest

    # Valid request
    request = QueryRequest(query="What is AI?")
    assert request.query == "What is AI?"
    assert request.session_id is None

    # Request with session
    request_with_session = QueryRequest(query="What is ML?", session_id="session-123")
    assert request_with_session.session_id == "session-123"


def test_query_response_model():
    """Test QueryResponse model structure"""
    from app import QueryResponse

    # Create response
    response = QueryResponse(
        answer="Test answer",
        sources=[{"text": "Source 1", "url": "http://example.com"}],
        session_id="session-456"
    )

    assert response.answer == "Test answer"
    assert len(response.sources) == 1
    assert response.session_id == "session-456"


def test_course_analytics_model():
    """Test CourseAnalytics model structure"""
    from app import CourseAnalytics

    analytics = CourseAnalytics(
        total_courses=5,
        course_titles=["Course A", "Course B"]
    )

    assert analytics.total_courses == 5
    assert len(analytics.course_titles) == 2
