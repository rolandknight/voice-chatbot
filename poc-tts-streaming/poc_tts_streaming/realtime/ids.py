"""Identifier and clock helpers shaped like the OpenAI Realtime API's."""

from __future__ import annotations

import secrets
import time


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def now() -> int:
    return int(time.time())
