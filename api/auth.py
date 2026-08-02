"""Simple API-key authentication.

Not a substitute for real OAuth/JWT-based auth in a multi-tenant product,
but it's the minimum a public API needs on day one: nothing should be
callable with zero credentials. Set API_KEY in the environment in any real
deployment -- the default here exists only so local development works out
of the box.
"""
from __future__ import annotations
import os

from fastapi import Header, HTTPException, status

API_KEY = os.environ.get("API_KEY", "dev-key")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
