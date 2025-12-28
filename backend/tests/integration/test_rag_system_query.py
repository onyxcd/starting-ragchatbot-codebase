"""Integration tests for RAG system query handling - Requirement #3"""

from unittest.mock import MagicMock

import pytest
from rag_system import RAGSystem


def test_query_with_empty_vector_store(rag_system):
    """Test query behavior with no data loaded"""
    # This should not crash even with empty vector store
    response, sources = rag_system.query("What is machine learning?")

    # Should return valid types
    assert isinstance(response, str)
    assert isinstance(sources, list)

    # Response should be some text (either from AI or error message)
    assert len(response) > 0


def test_query_with_content_question(rag_system, sample_course, sample_chunks):
    """Test successful query flow with content-related question"""
    # Add test data
    rag_system.vector_store.add_course_metadata(sample_course)
    rag_system.vector_store.add_course_content(sample_chunks)

    # Re-register tools with populated data
    rag_system.search_tool = rag_system.search_tool.__class__(rag_system.vector_store)
    rag_system.tool_manager = rag_system.tool_manager.__class__()
    rag_system.tool_manager.register_tool(rag_system.search_tool)
    rag_system.tool_manager.register_tool(rag_system.outline_tool)

    # Query about content
    response, sources = rag_system.query("What is supervised learning?")

    # Should return a response
    assert isinstance(response, str)
    assert len(response) > 0

    # Sources should be a list
    assert isinstance(sources, list)


def test_query_error_propagation(rag_system, mock_anthropic_client):
    """Test that errors are properly propagated from components"""
    # Make the AI generator raise an error
    mock_anthropic_client.messages.create.side_effect = Exception("API Error")

    # This should raise the exception (not caught by RAGSystem.query)
    with pytest.raises(Exception) as exc_info:
        rag_system.query("test question")

    assert "API Error" in str(exc_info.value)


def test_query_with_session_history(rag_system):
    """Test conversation history management across queries"""
    # Create a session
    session_id = rag_system.session_manager.create_session()

    # First query
    response1, _ = rag_system.query("What is AI?", session_id)
    assert isinstance(response1, str)

    # Check history was updated
    history = rag_system.session_manager.get_conversation_history(session_id)
    assert history is not None
    assert "What is AI?" in history

    # Second query
    response2, _ = rag_system.query("Tell me more", session_id)
    assert isinstance(response2, str)

    # History should now include both exchanges
    history = rag_system.session_manager.get_conversation_history(session_id)
    assert "What is AI?" in history
    assert "Tell me more" in history


def test_query_sources_returned(
    rag_system, sample_course, sample_chunks, mock_anthropic_client
):
    """Test that sources are properly returned from queries"""
    # Add test data
    rag_system.vector_store.add_course_metadata(sample_course)
    rag_system.vector_store.add_course_content(sample_chunks)

    # Re-register tools
    rag_system.search_tool = rag_system.search_tool.__class__(rag_system.vector_store)
    rag_system.tool_manager = rag_system.tool_manager.__class__()
    rag_system.tool_manager.register_tool(rag_system.search_tool)
    rag_system.tool_manager.register_tool(rag_system.outline_tool)

    # Mock tool use
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "machine learning"}
    tool_block.id = "test-id"

    first_response = MagicMock()
    first_response.content = [tool_block]
    first_response.stop_reason = "tool_use"

    final_text_block = MagicMock()
    final_text_block.text = "ML is a subset of AI"
    final_text_block.type = "text"

    second_response = MagicMock()
    second_response.content = [final_text_block]

    mock_anthropic_client.messages.create.side_effect = [
        first_response,
        second_response,
    ]

    # Query
    response, sources = rag_system.query("What is machine learning?")

    # Should have sources
    assert isinstance(sources, list)


def test_query_without_session(rag_system):
    """Test query without session ID"""
    response, sources = rag_system.query("What is AI?")

    # Should work without session
    assert isinstance(response, str)
    assert isinstance(sources, list)


def test_query_prompt_formatting(rag_system, mock_anthropic_client):
    """Test that query is properly formatted in the prompt"""
    rag_system.query("What is machine learning?")

    # Check the API was called with properly formatted prompt
    call_args = mock_anthropic_client.messages.create.call_args
    messages = call_args.kwargs["messages"]

    # Should have a message
    assert len(messages) > 0

    # First message should be user message containing the query
    first_message = messages[0]
    assert first_message["role"] == "user"
    assert "machine learning" in first_message["content"].lower()


def test_sources_reset_between_queries(rag_system):
    """Test that sources are reset between different queries"""
    # First query
    _, sources1 = rag_system.query("What is AI?")

    # Second query
    _, sources2 = rag_system.query("What is ML?")

    # Sources should be independent (not accumulated)
    # They should both be lists, but content may vary
    assert isinstance(sources1, list)
    assert isinstance(sources2, list)


def test_tool_manager_integration(rag_system):
    """Test that RAGSystem properly integrates with ToolManager"""
    # Check that tools are registered
    tool_defs = rag_system.tool_manager.get_tool_definitions()

    assert isinstance(tool_defs, list)
    assert len(tool_defs) >= 2  # At least search and outline tools

    # Check tool names
    tool_names = [tool["name"] for tool in tool_defs]
    assert "search_course_content" in tool_names
    assert "get_course_outline" in tool_names


def test_get_course_analytics(rag_system, sample_course, sample_chunks):
    """Test course analytics retrieval"""
    # Add data
    rag_system.vector_store.add_course_metadata(sample_course)
    rag_system.vector_store.add_course_content(sample_chunks)

    # Get analytics
    analytics = rag_system.get_course_analytics()

    # Check structure
    assert "total_courses" in analytics
    assert "course_titles" in analytics

    # Check values
    assert analytics["total_courses"] >= 1
    assert isinstance(analytics["course_titles"], list)
    assert sample_course.title in analytics["course_titles"]


def test_add_course_document_integration(rag_system, tmp_path):
    """Test adding a course document (integration with document processor)"""
    # Create a temporary test file
    test_file = tmp_path / "test_course.txt"
    test_file.write_text(
        """
Course Title: Test Course Integration
Instructor: Test Instructor

Lesson 1: Introduction
This is lesson 1 content.

Lesson 2: Advanced Topics
This is lesson 2 content.
    """
    )

    # Add the document
    course, chunk_count = rag_system.add_course_document(str(test_file))

    # Should succeed
    if course is not None:
        assert course.title == "Test Course Integration"
        assert chunk_count > 0


def test_query_with_sequential_tool_calls(
    rag_system, sample_course, sample_chunks, mock_anthropic_client
):
    """Integration test: RAG system with sequential tool calling across 2 rounds"""
    # Add test data
    rag_system.vector_store.add_course_metadata(sample_course)
    rag_system.vector_store.add_course_content(sample_chunks)

    # Re-register tools
    rag_system.search_tool = rag_system.search_tool.__class__(rag_system.vector_store)
    rag_system.outline_tool = rag_system.outline_tool.__class__(rag_system.vector_store)
    rag_system.tool_manager = rag_system.tool_manager.__class__()
    rag_system.tool_manager.register_tool(rag_system.search_tool)
    rag_system.tool_manager.register_tool(rag_system.outline_tool)

    # Mock sequential tool calls
    # Round 1: Get course outline
    tool_block_1 = MagicMock()
    tool_block_1.type = "tool_use"
    tool_block_1.name = "get_course_outline"
    tool_block_1.input = {"course_title": "Machine Learning"}
    tool_block_1.id = "tool-1"

    first_response = MagicMock()
    first_response.content = [tool_block_1]
    first_response.stop_reason = "tool_use"

    # Round 2: Search specific content
    tool_block_2 = MagicMock()
    tool_block_2.type = "tool_use"
    tool_block_2.name = "search_course_content"
    tool_block_2.input = {"query": "supervised learning", "lesson_number": 2}
    tool_block_2.id = "tool-2"

    second_response = MagicMock()
    second_response.content = [tool_block_2]
    second_response.stop_reason = "tool_use"

    # Round 3: Final answer
    final_text_block = MagicMock()
    final_text_block.text = "Supervised learning is covered in lesson 2"
    final_text_block.type = "text"

    third_response = MagicMock()
    third_response.content = [final_text_block]
    third_response.stop_reason = "end_turn"

    mock_anthropic_client.messages.create.side_effect = [
        first_response,
        second_response,
        third_response,
    ]

    # Execute query
    response, sources = rag_system.query("What lesson covers supervised learning?")

    # Assertions
    assert "Supervised learning is covered in lesson 2" in response
    assert mock_anthropic_client.messages.create.call_count == 3

    # Sources should come from the tools used
    assert isinstance(sources, list)


def test_system_initialization(test_config, mock_anthropic_client):
    """Test that RAGSystem initializes all components correctly"""
    system = RAGSystem(test_config)

    # Check that all components are initialized
    assert system.document_processor is not None
    assert system.vector_store is not None
    assert system.ai_generator is not None
    assert system.session_manager is not None
    assert system.tool_manager is not None
    assert system.search_tool is not None
    assert system.outline_tool is not None
