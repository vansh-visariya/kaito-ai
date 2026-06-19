"""Shared utility functions for Kaito-AI.

Provides helpers for ID generation.
"""

import uuid


def generate_unique_id() -> str:
    """Generate a random UUID4 string."""
    return str(uuid.uuid4())
