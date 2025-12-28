# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) chatbot system that answers questions about course materials. It combines semantic search (ChromaDB + Sentence Transformers) with AI-powered response generation (Claude API with tool calling) to provide intelligent, context-aware answers.

## Key Commands

### Setup
```bash
# Install dependencies
uv sync

# Set up environment (requires ANTHROPIC_API_KEY in .env file)
cp .env.example .env
# Then edit .env to add your API key
```

### Running the Application
```bash
# Quick start
./run.sh

# Or manually
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application serves at `http://localhost:8000` with API docs at `http://localhost:8000/docs`.

### Code Quality Tools
```bash
# Auto-format code with Black and Ruff
./format.sh

# Run all quality checks (formatting, linting, type checking, tests)
./quality.sh

# Individual tools
uv run black backend/                    # Format code
uv run ruff check backend/ --fix         # Lint and auto-fix
uv run mypy backend/                     # Type checking
uv run pytest backend/tests/             # Run tests
```

**Quality Tools Configured:**
- **Black** - Code formatter (line length: 88, Python 3.13+)
- **Ruff** - Fast Python linter (includes isort, pyflakes, pycodestyle, etc.)
- **mypy** - Static type checker
- **pytest** - Testing framework with coverage reporting

All tools are configured in `pyproject.toml` with consistent settings.

## Architecture

### Component Flow

The system is structured around a **tool-based AI architecture** where Claude autonomously decides when to search course materials:

```
User Query → FastAPI → RAGSystem → AIGenerator (Claude API)
                                         ↓
                                    Tool Decision
                                         ↓
                              ToolManager executes CourseSearchTool
                                         ↓
                              VectorStore searches ChromaDB
                                         ↓
                              Results back to Claude
                                     ↓
                              Claude synthesizes answer
                                         ↓
                              Response + Sources → User
```

### Core Components (backend/)

1. **app.py** - FastAPI entry point
   - Routes: `/api/query` (POST), `/api/courses` (GET)
   - Startup event loads documents from `../docs`
   - Serves frontend static files

2. **rag_system.py** - Main orchestrator
   - Coordinates all components
   - Manages document ingestion and query processing
   - **Key method**: `query(query, session_id)` - processes user queries using tool-based AI

3. **ai_generator.py** - Claude API integration
   - **Tool-based approach**: Claude decides when to search via tool calling
   - Handles tool execution loop (initial response → tool call → final response)
   - Static system prompt defines search tool usage protocol
   - Temperature: 0 (deterministic), Max tokens: 800

4. **search_tools.py** - Tool definitions and management
   - `CourseSearchTool`: Defines search tool for Claude with parameters (query, course_name, lesson_number)
   - `ToolManager`: Registers tools and executes them
   - **Important**: Tools track sources for UI display via `last_sources` attribute

5. **vector_store.py** - ChromaDB integration
   - **Two collections**:
     - `course_catalog`: Course metadata for semantic course name matching
     - `course_content`: Actual course chunks with embeddings
   - Embedding model: `all-MiniLM-L6-v2`
   - Unified search interface with fuzzy course name matching

6. **document_processor.py** - Document parsing
   - Supports PDF, DOCX, TXT
   - Extracts structured metadata (course title, instructor, lessons with links)
   - Chunks text (800 chars, 100 char overlap)
   - Creates `Course` and `CourseChunk` objects

7. **session_manager.py** - Conversation history
   - Maintains context across queries per session
   - Default: stores last 2 exchanges (configurable via `config.MAX_HISTORY`)

### Configuration (config.py)

All system parameters are centralized in `config.py`:
- `ANTHROPIC_MODEL`: "claude-sonnet-4-20250514"
- `EMBEDDING_MODEL`: "all-MiniLM-L6-v2"
- `CHUNK_SIZE`: 800 chars
- `CHUNK_OVERLAP`: 100 chars
- `MAX_RESULTS`: 5 search results
- `MAX_HISTORY`: 2 conversation exchanges
- `CHROMA_PATH`: "./chroma_db"

### Data Models (models.py)

Uses Pydantic models:
- `Course`: Represents a course with title, instructor, lessons list
- `Lesson`: Individual lesson with number, title, link
- `CourseChunk`: Text chunk with metadata (course title, lesson number)

## Important Architectural Patterns

### Tool-Based AI Flow

The system uses **Claude's tool calling** rather than explicit retrieval-then-generate:

1. Claude receives query with `search_course_content` tool available
2. Claude decides autonomously whether to search and what parameters to use
3. Tool execution happens via `_handle_tool_execution()` in `ai_generator.py`
4. Results are passed back to Claude in a second API call for synthesis

**Why this matters**: Don't manually call vector store searches before AI generation. Let Claude decide via tools.

### Two-Collection Vector Store Design

- **course_catalog**: Used for semantic course name matching (fuzzy search)
- **course_content**: Stores actual content chunks

When searching with a course name filter, the system:
1. Searches `course_catalog` to find the best matching course title
2. Uses that matched title to filter `course_content` search

This allows users to search with partial/fuzzy course names like "MCP" or "Computer Use".

### Duplicate Prevention

`add_course_folder()` in `rag_system.py` checks existing course titles to avoid re-processing documents. Uses `vector_store.get_existing_course_titles()` to get already loaded courses.

### Source Tracking

Sources are tracked via `last_sources` attribute on tools (not from vector store directly) because tool execution determines what was actually used in the response. The flow:
1. Tool executes search and stores sources in `self.last_sources`
2. After AI response, `tool_manager.get_last_sources()` retrieves them
3. Sources passed to frontend for display
4. Sources reset via `tool_manager.reset_sources()` before next query

## Development Notes

### Using uv Commands

**IMPORTANT**: Always use `uv run` for executing commands (e.g., `uv run uvicorn`, `uv run pytest`). Never invoke `uv` directly without the `run` subcommand. Use the `./run.sh` script for quick starts.

### Adding New Document Types

Extend `document_processor.py` to handle additional file types. Current parsing logic expects specific metadata format in documents (course title, instructor, lesson markers).

### Modifying Search Parameters

All search tuning happens in two places:
- `config.py`: Global defaults (MAX_RESULTS, CHUNK_SIZE, etc.)
- `vector_store.py`: Search implementation details (similarity thresholds, etc.)

### Adding New Tools

To add a new tool for Claude:
1. Create a class extending `Tool` in `search_tools.py`
2. Implement `get_tool_definition()` and `execute()`
3. Register it in `RAGSystem.__init__()` via `tool_manager.register_tool()`

### ChromaDB Persistence

ChromaDB data persists in `backend/chroma_db/`. To reset:
```bash
rm -rf backend/chroma_db/
```
Documents will reload on next startup from `docs/` folder.

### Frontend Integration

Frontend is vanilla JS (no build process). Located in `frontend/`:
- Static files served by FastAPI at root `/`
- Uses fetch API to call `/api/query` and `/api/courses`
- Marked.js renders markdown responses

## Environment Requirements

- Python 3.13+ (specified in `.python-version`)
- `uv` package manager
- Anthropic API key in `.env`
- For Windows: Use Git Bash to run commands
