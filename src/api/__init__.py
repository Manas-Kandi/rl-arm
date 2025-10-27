"""FastAPI backend package for Panda Door RL monitoring."""

from typing import Any

__all__ = ["create_app"]


def create_app(*args: Any, **kwargs: Any):
    from .server import create_app as _create_app

    return _create_app(*args, **kwargs)
