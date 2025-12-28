#!/bin/bash
# Code Quality Check Script
# Runs all quality checks for the RAG chatbot codebase

set -e  # Exit on first error

echo "🔍 Running code quality checks..."
echo ""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check 1: Black formatting
echo -e "${BLUE}1. Checking code formatting with Black...${NC}"
if uv run black --check backend/; then
    echo -e "${GREEN}✓ Black formatting check passed${NC}"
else
    echo "❌ Black formatting check failed"
    echo "Run: uv run black backend/ to fix formatting"
    exit 1
fi
echo ""

# Check 2: Ruff linting
echo -e "${BLUE}2. Running Ruff linter...${NC}"
if uv run ruff check backend/; then
    echo -e "${GREEN}✓ Ruff linting passed${NC}"
else
    echo "❌ Ruff linting failed"
    echo "Run: uv run ruff check backend/ --fix to auto-fix issues"
    exit 1
fi
echo ""

# Check 3: Type checking with mypy
echo -e "${BLUE}3. Running mypy type checker...${NC}"
if uv run mypy backend/ --exclude backend/tests/; then
    echo -e "${GREEN}✓ Type checking passed${NC}"
else
    echo "⚠️  Type checking found issues (non-blocking)"
fi
echo ""

# Check 4: Run tests
echo -e "${BLUE}4. Running test suite...${NC}"
if uv run pytest backend/tests/ -v; then
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo "❌ Tests failed"
    exit 1
fi
echo ""

echo -e "${GREEN}✨ All quality checks completed successfully!${NC}"
