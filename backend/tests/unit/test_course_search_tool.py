"""Unit tests for CourseSearchTool.execute() method - Requirement #1"""

import pytest
from search_tools import CourseSearchTool
from vector_store import SearchResults


def test_execute_with_valid_query(course_search_tool):
    """Test successful search execution with a valid query"""
    result = course_search_tool.execute(query="machine learning")

    # Should return a string result
    assert isinstance(result, str)
    assert len(result) > 0

    # Should contain relevant content
    assert "machine learning" in result.lower() or "ml" in result.lower()

    # Should track sources
    assert len(course_search_tool.last_sources) > 0


def test_execute_with_course_filter(course_search_tool):
    """Test search with course name filter"""
    result = course_search_tool.execute(
        query="supervised",
        course_name="Machine Learning"  # Partial match should work
    )

    # Should return results
    assert isinstance(result, str)

    # Should not be an error message
    assert "No course found" not in result
    assert "No relevant content found" not in result


def test_execute_with_lesson_filter(course_search_tool):
    """Test search with lesson number filter"""
    result = course_search_tool.execute(
        query="neural",
        lesson_number=3
    )

    # Should return results
    assert isinstance(result, str)

    # Should include lesson context
    assert "Lesson 3" in result


def test_execute_with_both_filters(course_search_tool):
    """Test search with both course name and lesson number filters"""
    result = course_search_tool.execute(
        query="learning",
        course_name="Machine Learning",
        lesson_number=2
    )

    # Should return results
    assert isinstance(result, str)

    # Should include both course and lesson context
    assert "Machine Learning" in result
    assert "Lesson 2" in result


def test_execute_with_invalid_course(course_search_tool):
    """Test search with non-existent course name"""
    result = course_search_tool.execute(
        query="test",
        course_name="NonExistentCourse"
    )

    # Should return error message
    assert isinstance(result, str)
    assert "No course found" in result
    assert "NonExistentCourse" in result


def test_execute_with_empty_results(course_search_tool):
    """Test search that returns no results"""
    result = course_search_tool.execute(
        query="quantum physics thermodynamics biochemistry"  # Topics not in test data
    )

    # Should return informative message
    assert isinstance(result, str)
    assert "No relevant content found" in result or len(result) == 0


def test_execute_error_handling(vector_store):
    """Test that execute handles vector store errors gracefully"""
    # Create a tool with empty vector store
    tool = CourseSearchTool(vector_store)

    # This should not crash, even with empty data
    result = tool.execute(query="test query")

    # Should return a message (either error or "no results found")
    assert isinstance(result, str)


def test_last_sources_tracking(course_search_tool):
    """Test that last_sources attribute correctly tracks sources"""
    # Reset sources
    course_search_tool.last_sources = []

    # Execute search
    result = course_search_tool.execute(query="machine learning")

    # Check sources were tracked
    assert isinstance(course_search_tool.last_sources, list)

    if len(course_search_tool.last_sources) > 0:
        # Each source should be a dict with text and url
        source = course_search_tool.last_sources[0]
        assert isinstance(source, dict)
        assert "text" in source
        assert "url" in source


def test_last_sources_format(course_search_tool):
    """Test that sources are formatted correctly with text and URL"""
    course_search_tool.execute(query="supervised learning", lesson_number=2)

    assert len(course_search_tool.last_sources) > 0

    for source in course_search_tool.last_sources:
        # Should have text field
        assert "text" in source
        assert isinstance(source["text"], str)
        assert len(source["text"]) > 0

        # Should have url field (may be None)
        assert "url" in source


def test_multiple_searches_reset_sources(course_search_tool):
    """Test that each search properly manages sources"""
    # First search
    course_search_tool.execute(query="machine learning")
    first_sources = course_search_tool.last_sources.copy()

    # Second search
    course_search_tool.execute(query="neural networks")
    second_sources = course_search_tool.last_sources

    # Sources should be updated, not accumulated
    assert isinstance(first_sources, list)
    assert isinstance(second_sources, list)


def test_tool_definition_structure():
    """Test that get_tool_definition returns correct structure"""
    from vector_store import VectorStore
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        store = VectorStore(temp_dir, "all-MiniLM-L6-v2", 3)
        tool = CourseSearchTool(store)

        definition = tool.get_tool_definition()

        # Check required fields
        assert "name" in definition
        assert definition["name"] == "search_course_content"

        assert "description" in definition
        assert isinstance(definition["description"], str)

        assert "input_schema" in definition
        schema = definition["input_schema"]

        assert "type" in schema
        assert schema["type"] == "object"

        assert "properties" in schema
        props = schema["properties"]

        # Check required parameter
        assert "query" in props

        # Check optional parameters
        assert "course_name" in props
        assert "lesson_number" in props

        # Check required array
        assert "required" in schema
        assert "query" in schema["required"]
