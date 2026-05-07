#!/usr/bin/env python3
"""
Script to rebuild the knowledge base with the updated persona engine.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_pipeline
from src.logger import logger

if __name__ == "__main__":
    logger.info("Starting KB rebuild with updated persona engine...")
    print("\n" + "="*70)
    print("REBUILDING KNOWLEDGE BASE WITH NEW PERSONA ENGINE")
    print("  - Semantic Frame Engine")
    print("  - Entity-scoped Sentiment Scoper")
    print("="*70 + "\n")
    
    try:
        run_pipeline(force_rebuild=True)
        print("\n" + "="*70)
        print("✓ KB REBUILD COMPLETE")
        print("="*70 + "\n")
        logger.info("KB rebuild successful!")
    except Exception as e:
        logger.error(f"KB rebuild failed: {e}")
        print(f"\n✗ KB rebuild failed: {e}\n")
        sys.exit(1)
