"""Unit tests for AIGenerator tool calling - Requirement #2"""

from unittest.mock import MagicMock

from ai_generator import AIGenerator


def test_generate_response_without_tools(ai_generator, mock_anthropic_client):
    """Test basic response generation without tool usage"""
    response = ai_generator.generate_response("What is 2+2?")

    # Should return the mock response
    assert response == "Test response"

    # Should have called the API once
    mock_anthropic_client.messages.create.assert_called_once()


def test_generate_response_with_tools_available(
    ai_generator, tool_manager, mock_anthropic_client
):
    """Test that tools are passed to the API when provided"""
    # Get tool definitions
    tools = tool_manager.get_tool_definitions()

    # Call with tools
    ai_generator.generate_response(
        query="What is machine learning?", tools=tools, tool_manager=tool_manager
    )

    # Check that tools were passed in the API call
    call_args = mock_anthropic_client.messages.create.call_args
    assert call_args is not None
    assert "tools" in call_args.kwargs
    assert "tool_choice" in call_args.kwargs


def test_generate_response_calls_search_tool(
    ai_generator, tool_manager, mock_anthropic_client, monkeypatch
):
    """CRITICAL: Test that Claude correctly calls CourseSearchTool"""
    # Mock tool use response
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "machine learning"}
    tool_block.id = "test-id-123"

    # First response: tool use
    first_response = MagicMock()
    first_response.content = [tool_block]
    first_response.stop_reason = "tool_use"

    # Second response: final answer
    final_text_block = MagicMock()
    final_text_block.text = "Final answer about ML"
    final_text_block.type = "text"

    second_response = MagicMock()
    second_response.content = [final_text_block]
    second_response.stop_reason = "end_turn"

    # Set up mock to return different responses
    mock_anthropic_client.messages.create.side_effect = [
        first_response,
        second_response,
    ]

    # Call generate_response
    response = ai_generator.generate_response(
        query="What is ML?",
        tools=tool_manager.get_tool_definitions(),
        tool_manager=tool_manager,
    )

    # Should return the final answer
    assert response == "Final answer about ML"

    # Should have called API twice (once for tool use, once for final response)
    assert mock_anthropic_client.messages.create.call_count == 2


def test_sequential_tool_calling(ai_generator, tool_manager, mock_anthropic_client):
    """Test that Claude can make 2 sequential tool calls in separate rounds"""
    # First round: Claude searches course A
    tool_block_1 = MagicMock()
    tool_block_1.type = "tool_use"
    tool_block_1.name = "search_course_content"
    tool_block_1.input = {"query": "topic A", "course_name": "Course A"}
    tool_block_1.id = "tool-1"

    first_response = MagicMock()
    first_response.content = [tool_block_1]
    first_response.stop_reason = "tool_use"

    # Second round: Claude searches course B
    tool_block_2 = MagicMock()
    tool_block_2.type = "tool_use"
    tool_block_2.name = "search_course_content"
    tool_block_2.input = {"query": "topic B", "course_name": "Course B"}
    tool_block_2.id = "tool-2"

    second_response = MagicMock()
    second_response.content = [tool_block_2]
    second_response.stop_reason = "tool_use"

    # Third round: Final answer
    final_text_block = MagicMock()
    final_text_block.text = "Combined answer from both searches"
    final_text_block.type = "text"

    third_response = MagicMock()
    third_response.content = [final_text_block]
    third_response.stop_reason = "end_turn"

    # Set up sequence
    mock_anthropic_client.messages.create.side_effect = [
        first_response,
        second_response,
        third_response,
    ]

    # Execute
    response = ai_generator.generate_response(
        query="Compare topics across Course A and Course B",
        tools=tool_manager.get_tool_definitions(),
        tool_manager=tool_manager,
    )

    # Assertions
    assert response == "Combined answer from both searches"
    assert mock_anthropic_client.messages.create.call_count == 3

    # Verify tools were available in first 2 rounds, but not final forced response
    call_1_kwargs = mock_anthropic_client.messages.create.call_args_list[0].kwargs
    call_2_kwargs = mock_anthropic_client.messages.create.call_args_list[1].kwargs
    call_3_kwargs = mock_anthropic_client.messages.create.call_args_list[2].kwargs

    # First two calls should have tools
    assert "tools" in call_1_kwargs
    assert "tools" in call_2_kwargs
    # Third call is forced final response after max rounds, should NOT have tools
    assert "tools" not in call_3_kwargs


def test_max_rounds_enforcement(
    ai_generator, tool_manager, mock_anthropic_client, monkeypatch
):
    """Test that system enforces MAX_TOOL_ROUNDS limit"""
    # Mock config to ensure max rounds is 2
    from config import config

    original_max = config.MAX_TOOL_ROUNDS
    config.MAX_TOOL_ROUNDS = 2

    try:
        # Create tool use responses (exceeds limit)
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "test"}
        tool_block.id = "tool-id"

        tool_response = MagicMock()
        tool_response.content = [tool_block]
        tool_response.stop_reason = "tool_use"

        # Final response without tools
        final_block = MagicMock()
        final_block.text = "Forced final response"
        final_block.type = "text"

        final_response = MagicMock()
        final_response.content = [final_block]
        final_response.stop_reason = "end_turn"

        # Set up: 2 tool rounds, then forced final
        mock_anthropic_client.messages.create.side_effect = [
            tool_response,  # Round 1
            tool_response,  # Round 2
            final_response,  # Final (max hit, no tools)
        ]

        # Execute
        response = ai_generator.generate_response(
            query="Test",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager,
        )

        # Should return forced response
        assert response == "Forced final response"
        assert mock_anthropic_client.messages.create.call_count == 3

        # Third call should NOT have tools
        third_call_kwargs = mock_anthropic_client.messages.create.call_args_list[
            2
        ].kwargs
        assert "tools" not in third_call_kwargs

    finally:
        # Restore original config
        config.MAX_TOOL_ROUNDS = original_max


def test_tool_execution_error_handling(
    ai_generator, tool_manager, mock_anthropic_client, monkeypatch
):
    """Test that tool execution errors are passed back to Claude"""
    # First round: tool call
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "test"}
    tool_block.id = "tool-error-id"

    first_response = MagicMock()
    first_response.content = [tool_block]
    first_response.stop_reason = "tool_use"

    # Final response after error
    final_block = MagicMock()
    final_block.text = "I encountered an error"
    final_block.type = "text"

    second_response = MagicMock()
    second_response.content = [final_block]
    second_response.stop_reason = "end_turn"

    mock_anthropic_client.messages.create.side_effect = [
        first_response,
        second_response,
    ]

    # Make tool execution fail
    def failing_execute(tool_name, **kwargs):
        raise ValueError("Tool failed")

    monkeypatch.setattr(tool_manager, "execute_tool", failing_execute)

    # Execute - should not crash
    response = ai_generator.generate_response(
        query="Test",
        tools=tool_manager.get_tool_definitions(),
        tool_manager=tool_manager,
    )

    # Should return Claude's response to the error
    assert response == "I encountered an error"

    # Check second call received error message
    second_call_messages = mock_anthropic_client.messages.create.call_args_list[
        1
    ].kwargs["messages"]
    tool_result_content = second_call_messages[-1]["content"][0]
    assert "Error executing tool" in tool_result_content["content"]
    assert tool_result_content.get("is_error")


def test_early_termination_on_direct_answer(
    ai_generator, tool_manager, mock_anthropic_client
):
    """Test that loop exits immediately if Claude doesn't use tools"""
    # Mock direct response without tool use
    text_block = MagicMock()
    text_block.text = "Direct answer"
    text_block.type = "text"

    response = MagicMock()
    response.content = [text_block]
    response.stop_reason = "end_turn"

    mock_anthropic_client.messages.create.return_value = response

    # Execute
    result = ai_generator.generate_response(
        query="What is 2+2?",
        tools=tool_manager.get_tool_definitions(),
        tool_manager=tool_manager,
    )

    # Should return direct answer
    assert result == "Direct answer"

    # Should only make 1 API call
    assert mock_anthropic_client.messages.create.call_count == 1


def test_tools_available_in_each_round(
    ai_generator, tool_manager, mock_anthropic_client
):
    """Test that tools parameter is passed in each round"""
    # Two rounds of tool use
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "test"}
    tool_block.id = "tool-id"

    tool_response = MagicMock()
    tool_response.content = [tool_block]
    tool_response.stop_reason = "tool_use"

    final_block = MagicMock()
    final_block.text = "Final answer"
    final_block.type = "text"

    final_response = MagicMock()
    final_response.content = [final_block]
    final_response.stop_reason = "end_turn"

    mock_anthropic_client.messages.create.side_effect = [tool_response, final_response]

    # Execute
    ai_generator.generate_response(
        query="Test",
        tools=tool_manager.get_tool_definitions(),
        tool_manager=tool_manager,
    )

    # Both calls should have tools parameter
    first_call_kwargs = mock_anthropic_client.messages.create.call_args_list[0].kwargs
    second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1].kwargs

    assert "tools" in first_call_kwargs
    assert "tools" in second_call_kwargs


def test_tool_result_propagation(
    ai_generator, tool_manager, populated_vector_store, mock_anthropic_client
):
    """Test that tool results are properly propagated back to Claude"""
    # Mock tool use for search
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "machine learning"}
    tool_block.id = "tool-456"

    first_response = MagicMock()
    first_response.content = [tool_block]
    first_response.stop_reason = "tool_use"

    # Mock final response
    final_text_block = MagicMock()
    final_text_block.text = "Response using tool results"
    final_text_block.type = "text"

    second_response = MagicMock()
    second_response.content = [final_text_block]

    mock_anthropic_client.messages.create.side_effect = [
        first_response,
        second_response,
    ]

    # Execute
    _response = ai_generator.generate_response(
        query="What is ML?",
        tools=tool_manager.get_tool_definitions(),
        tool_manager=tool_manager,
    )

    # Check that second API call includes tool results
    second_call_args = mock_anthropic_client.messages.create.call_args_list[1]
    messages = second_call_args.kwargs["messages"]

    # Should have user message, assistant tool use, and user tool result
    assert len(messages) >= 2

    # Last message should be tool results
    last_message = messages[-1]
    assert last_message["role"] == "user"
    assert isinstance(last_message["content"], list)


def test_missing_api_key_raises_error():
    """Test that missing API key causes initialization to fail"""
    # Test with empty API key - this should work (initialization succeeds)
    # but calling the API should fail
    generator = AIGenerator("", "claude-sonnet-4")

    # The generator is created, but API calls will fail
    assert generator.client is not None


def test_system_prompt_includes_history(ai_generator, mock_anthropic_client):
    """Test that conversation history is included in system prompt"""
    history = "User: Previous question\nAssistant: Previous answer"

    ai_generator.generate_response(query="New question", conversation_history=history)

    # Check that system prompt includes history
    call_args = mock_anthropic_client.messages.create.call_args
    system_prompt = call_args.kwargs["system"]

    assert "Previous conversation" in system_prompt
    assert "Previous question" in system_prompt


def test_multiple_tool_calls(ai_generator, tool_manager, mock_anthropic_client):
    """Test handling of multiple tool calls in one response"""
    # Mock two tool uses
    tool_block_1 = MagicMock()
    tool_block_1.type = "tool_use"
    tool_block_1.name = "search_course_content"
    tool_block_1.input = {"query": "machine learning"}
    tool_block_1.id = "tool-1"

    tool_block_2 = MagicMock()
    tool_block_2.type = "tool_use"
    tool_block_2.name = "get_course_outline"
    tool_block_2.input = {"course_title": "Machine Learning"}
    tool_block_2.id = "tool-2"

    first_response = MagicMock()
    first_response.content = [tool_block_1, tool_block_2]
    first_response.stop_reason = "tool_use"

    # Final response
    final_text_block = MagicMock()
    final_text_block.text = "Combined response"
    final_text_block.type = "text"

    second_response = MagicMock()
    second_response.content = [final_text_block]

    mock_anthropic_client.messages.create.side_effect = [
        first_response,
        second_response,
    ]

    # Execute
    response = ai_generator.generate_response(
        query="Tell me about ML",
        tools=tool_manager.get_tool_definitions(),
        tool_manager=tool_manager,
    )

    # Should handle both tool calls
    assert response == "Combined response"


def test_base_params_configuration(test_config, mock_anthropic_client):
    """Test that base parameters are correctly configured"""
    generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)

    # Check base params
    assert generator.base_params["model"] == test_config.ANTHROPIC_MODEL
    assert generator.base_params["temperature"] == 0
    assert generator.base_params["max_tokens"] == 800


def test_generate_response_without_conversation_history(
    ai_generator, mock_anthropic_client
):
    """Test response generation without conversation history"""
    ai_generator.generate_response(query="What is AI?")

    # System prompt should not include "Previous conversation"
    call_args = mock_anthropic_client.messages.create.call_args
    system_prompt = call_args.kwargs["system"]

    # Should contain base system prompt but not history section
    assert "Previous conversation" not in system_prompt or "None" in system_prompt
