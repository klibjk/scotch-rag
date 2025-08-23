"""
Scotch-RAG: Frontend Package

This package contains the FastHTML + MonsterUI frontend implementation
for the Scotch-RAG chat interface.
"""

from .app import create_app

__version__ = "1.0.0"
__all__ = ["create_app"]
