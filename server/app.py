"""
server/app.py — Entry point for OpenEnv multi-mode deployment.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from server import app  # noqa: F401

__all__ = ["app"]


def main():
    """Main entry point for OpenEnv multi-mode deployment."""
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
