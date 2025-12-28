#!/bin/bash
# Auto-format code with Black and Ruff
# This script automatically fixes formatting and linting issues

echo "🔧 Formatting code..."
echo ""

# Format with Black
echo "Running Black formatter..."
uv run black backend/
echo "✓ Black formatting complete"
echo ""

# Fix auto-fixable Ruff issues
echo "Running Ruff auto-fixes..."
uv run ruff check backend/ --fix
echo "✓ Ruff fixes applied"
echo ""

echo "✨ Code formatting complete!"
echo "Run ./quality.sh to verify all checks pass"
