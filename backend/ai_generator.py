import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools.

Tool Usage Guidelines:
- **get_course_outline**: Use when users ask about:
  - What lessons/topics a course covers
  - Course structure or outline
  - List of lessons in a course
  - Overview of course content
- **search_course_content**: Use when users ask about:
  - Specific concepts or topics within course materials
  - Detailed explanations from lessons
  - Questions requiring searching through actual course content
- **Multi-round tool calling**: You may call tools across up to 2 rounds to gather information
  - Each round is separate - after seeing tool results, you can decide to call tools again or provide final answer
  - Use multiple rounds for complex queries requiring different searches or when combining information from different courses/lessons
  - Examples: Round 1: Get course outline → Round 2: Search specific lesson content → Final answer
- Synthesize tool results into accurate, fact-based responses
- If tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course structure questions**: Use get_course_outline tool first, then answer
- **Content-specific questions**: Use search_course_content tool first, then answer
- **No meta-commentary**:
  - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
  - Do not mention "based on the tool results" or "after searching"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with multi-round tool usage and conversation context.

        Supports up to MAX_TOOL_ROUNDS sequential tool calling rounds where Claude
        can reason about previous tool results and make additional tool calls.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        from config import config

        # Build system content efficiently
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize conversation messages
        messages = [{"role": "user", "content": query}]

        # Track rounds
        current_round = 0

        # Main tool execution loop
        while current_round < config.MAX_TOOL_ROUNDS:
            current_round += 1

            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }

            # Add tools if available (keep tools in all rounds)
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Make API call
            try:
                response = self.client.messages.create(**api_params)
            except Exception as e:
                # Re-raise API errors to be handled by caller
                raise

            # Condition 1: Claude returned a final answer (no tool use)
            if response.stop_reason != "tool_use":
                # Extract and return the text response
                return self._extract_text_response(response)

            # Condition 2: Tool use detected
            if response.stop_reason == "tool_use":
                # Condition 3: No tool manager provided (error condition)
                if not tool_manager:
                    # Fallback: return whatever text we can extract
                    return self._extract_text_response(response)

                # Execute tools and prepare for next round
                try:
                    # Add assistant's tool use response to messages
                    messages.append({
                        "role": "assistant",
                        "content": response.content
                    })

                    # Execute all tool calls
                    tool_results = self._execute_all_tools(response, tool_manager)

                    # Condition 4: Tool execution failed
                    if not tool_results:
                        # No valid tool results, break the loop
                        return "I encountered an error while searching. Please try rephrasing your question."

                    # Add tool results as user message
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })

                    # Continue to next round (loop continues)

                except Exception as e:
                    # Condition 5: Exception during tool execution
                    return f"I encountered an error while processing your request: {str(e)}"

        # Condition 6: Max rounds reached - make final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
            # Note: no tools parameter
        }

        try:
            final_response = self.client.messages.create(**final_params)
            return self._extract_text_response(final_response)
        except Exception as e:
            return "I've gathered information but encountered an error forming a response."

    def _extract_text_response(self, response) -> str:
        """
        Safely extract text content from Claude API response.

        Args:
            response: Claude API response object

        Returns:
            Text string from response
        """
        # Look for text content blocks
        for content_block in response.content:
            if hasattr(content_block, 'type') and content_block.type == "text":
                return content_block.text
            elif hasattr(content_block, 'text'):
                return content_block.text

        # Fallback: no text found
        return "I was unable to generate a response."

    def _execute_all_tools(self, response, tool_manager) -> Optional[List[Dict[str, Any]]]:
        """
        Execute all tool calls from an API response.

        Args:
            response: Claude API response containing tool_use blocks
            tool_manager: Manager to execute tools

        Returns:
            List of tool result dictionaries formatted for Claude API, or None if no results
        """
        tool_results = []

        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    # Execute the tool
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )

                    # Add successful result
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })

                except Exception as e:
                    # Add error result for this specific tool
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Error executing tool: {str(e)}",
                        "is_error": True
                    })

        return tool_results if tool_results else None