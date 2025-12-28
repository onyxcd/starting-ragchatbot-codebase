#!/usr/bin/env python3
"""Quick test to verify API key validation works"""

import sys
from config import Config
from rag_system import RAGSystem

try:
    print("Testing RAGSystem initialization with current .env...")
    config = Config()
    print(f"API key loaded: {config.ANTHROPIC_API_KEY[:20]}...")

    system = RAGSystem(config)
    print("❌ UNEXPECTED: RAGSystem initialized successfully (should have failed with placeholder)")

except ValueError as e:
    print(f"✅ SUCCESS: Validation caught the placeholder key!")
    print(f"\nError message:\n{e}")

except Exception as e:
    print(f"❌ UNEXPECTED ERROR: {type(e).__name__}: {e}")
