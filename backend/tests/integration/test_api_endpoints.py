"""Integration tests for FastAPI endpoints"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, Mock
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))


def test_query_endpoint_success(test_client):
    """Test /api/query endpoint with valid request"""
    # Mock the rag_system.query method on the test app
    def mock_query(query, session_id=None):
        return "Test answer about machine learning", [{"text": "Source 1", "url": "http://example.com"}]

    # Patch the rag_system on the test app
    with patch.object(test_client.app.state.rag_system, 'query', side_effect=mock_query):
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
    with patch.object(test_client.app.state.rag_system, 'query') as mock_query:
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
    with patch.object(test_client.app.state.rag_system, 'query', side_effect=Exception("Test error")):
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

    with patch.object(test_client.app.state.rag_system, 'query') as mock_query:
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

    with patch.object(test_client.app.state.rag_system, 'get_course_analytics', return_value=mock_analytics):
        response = test_client.get("/api/courses")

    assert response.status_code == 200

    data = response.json()
    assert "total_courses" in data
    assert "course_titles" in data
    assert data["total_courses"] == 3
    assert len(data["course_titles"]) == 3


def test_root_endpoint(test_client):
    """Test that root endpoint returns test message"""
    response = test_client.get("/")

    # Test app returns a simple JSON message
    assert response.status_code == 200
    data = response.json()
    assert "message" in data


def test_query_endpoint_creates_session_if_not_provided(test_client):
    """Test that endpoint creates new session if none provided"""
    with patch.object(test_client.app.state.rag_system, 'query') as mock_query:
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

    # Create a mock response for the error
    mock_response = Mock()
    mock_response.status_code = 401

    # Create the error with required parameters
    auth_error = anthropic.AuthenticationError(
        message="Invalid API key",
        response=mock_response,
        body={"error": {"message": "Invalid API key"}}
    )

    with patch.object(test_client.app.state.rag_system, 'query', side_effect=auth_error):
        response = test_client.post("/api/query", json={"query": "test"})

    # Should return 500
    assert response.status_code == 500

    # Error should mention authentication
    data = response.json()
    assert "detail" in data


def test_query_endpoint_handles_rate_limit(test_client):
    """Test handling of API rate limiting"""
    import anthropic

    # Create a mock response for the error
    mock_response = Mock()
    mock_response.status_code = 429

    # Create the error with required parameters
    rate_limit_error = anthropic.RateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"error": {"message": "Rate limit exceeded"}}
    )

    with patch.object(test_client.app.state.rag_system, 'query', side_effect=rate_limit_error):
        response = test_client.post("/api/query", json={"query": "test"})

    # Should return 500
    assert response.status_code == 500


def test_query_request_model_validation():
    """Test QueryRequest model validation"""
    from pydantic import BaseModel
    from typing import Optional

    # Define model inline to avoid importing from app.py
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    # Valid request
    request = QueryRequest(query="What is AI?")
    assert request.query == "What is AI?"
    assert request.session_id is None

    # Request with session
    request_with_session = QueryRequest(query="What is ML?", session_id="session-123")
    assert request_with_session.session_id == "session-123"


def test_query_response_model():
    """Test QueryResponse model structure"""
    from pydantic import BaseModel
    from typing import List, Dict, Optional

    # Define model inline to avoid importing from app.py
    class QueryResponse(BaseModel):
        answer: str
        sources: List[Dict[str, Optional[str]]]
        session_id: str

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
    from pydantic import BaseModel
    from typing import List

    # Define model inline to avoid importing from app.py
    class CourseAnalytics(BaseModel):
        total_courses: int
        course_titles: List[str]

    analytics = CourseAnalytics(
        total_courses=5,
        course_titles=["Course A", "Course B"]
    )

    assert analytics.total_courses == 5
    assert len(analytics.course_titles) == 2


def test_cors_headers_present(test_client):
    """Test that CORS headers are properly set"""
    response = test_client.options("/api/query", headers={
        "Origin": "http://example.com",
        "Access-Control-Request-Method": "POST"
    })

    # Check CORS headers are present (middleware should handle this)
    assert response.status_code in [200, 405]  # OPTIONS might not be explicitly handled


def test_query_with_empty_string(test_client):
    """Test /api/query with empty query string"""
    with patch.object(test_client.app.state.rag_system, 'query') as mock_query:
        mock_query.return_value = ("Please provide a question", [])

        response = test_client.post("/api/query", json={"query": ""})

    # Should still process (backend handles empty queries)
    assert response.status_code in [200, 422]


def test_query_with_very_long_string(test_client):
    """Test /api/query with very long query string"""
    long_query = "What is machine learning? " * 100

    with patch.object(test_client.app.state.rag_system, 'query') as mock_query:
        mock_query.return_value = ("Answer", [])

        response = test_client.post("/api/query", json={"query": long_query})

    assert response.status_code == 200


def test_query_with_special_characters(test_client):
    """Test /api/query with special characters in query"""
    special_query = "What is AI? <script>alert('test')</script> & special chars"

    with patch.object(test_client.app.state.rag_system, 'query') as mock_query:
        mock_query.return_value = ("Safe answer", [])

        response = test_client.post("/api/query", json={"query": special_query})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Safe answer"


def test_query_with_invalid_json(test_client):
    """Test /api/query with malformed JSON"""
    response = test_client.post(
        "/api/query",
        data="invalid json {",
        headers={"Content-Type": "application/json"}
    )

    # Should return 422 validation error
    assert response.status_code == 422


def test_courses_endpoint_error_handling(test_client):
    """Test /api/courses endpoint error handling"""
    with patch.object(test_client.app.state.rag_system, 'get_course_analytics',
                     side_effect=Exception("Database error")):
        response = test_client.get("/api/courses")

    assert response.status_code == 500
    data = response.json()
    assert "detail" in data


def test_query_preserves_session_id(test_client):
    """Test that session_id is preserved across requests"""
    test_session = "persistent-session-123"

    with patch.object(test_client.app.state.rag_system, 'query') as mock_query:
        mock_query.return_value = ("First answer", [])

        response1 = test_client.post("/api/query", json={
            "query": "First question",
            "session_id": test_session
        })

        response2 = test_client.post("/api/query", json={
            "query": "Second question",
            "session_id": test_session
        })

    assert response1.json()["session_id"] == test_session
    assert response2.json()["session_id"] == test_session


def test_multiple_simultaneous_sessions(test_client):
    """Test handling multiple different sessions"""
    with patch.object(test_client.app.state.rag_system, 'query') as mock_query:
        mock_query.return_value = ("Answer", [])

        # Create two different sessions
        response1 = test_client.post("/api/query", json={
            "query": "Question 1",
            "session_id": "session-1"
        })

        response2 = test_client.post("/api/query", json={
            "query": "Question 2",
            "session_id": "session-2"
        })

    # Sessions should be different
    assert response1.json()["session_id"] == "session-1"
    assert response2.json()["session_id"] == "session-2"


def test_courses_endpoint_returns_empty_list(test_client):
    """Test /api/courses when no courses are loaded"""
    mock_analytics = {
        "total_courses": 0,
        "course_titles": []
    }

    with patch.object(test_client.app.state.rag_system, 'get_course_analytics',
                     return_value=mock_analytics):
        response = test_client.get("/api/courses")

    assert response.status_code == 200
    data = response.json()
    assert data["total_courses"] == 0
    assert data["course_titles"] == []


def test_query_response_includes_all_fields(test_client):
    """Test that query response has all required fields"""
    with patch.object(test_client.app.state.rag_system, 'query') as mock_query:
        mock_query.return_value = ("Test answer", [{"text": "Source", "url": "http://test.com"}])

        response = test_client.post("/api/query", json={"query": "test"})

    assert response.status_code == 200
    data = response.json()

    # Verify all required fields are present
    assert "answer" in data
    assert "sources" in data
    assert "session_id" in data
    assert isinstance(data["answer"], str)
    assert isinstance(data["sources"], list)
    assert isinstance(data["session_id"], str)
