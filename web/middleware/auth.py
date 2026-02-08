"""API authentication middleware using bearer tokens.

Auto-generates a token on first run, stored at ~/.shadowai/api_token.
All /api/* routes require Authorization: Bearer <token> except exempted paths.
Localhost requests and TESTING mode bypass auth.
"""

import logging
import os
import secrets
from pathlib import Path
from functools import wraps
from flask import request, jsonify, current_app

log = logging.getLogger(__name__)

TOKEN_FILE = os.path.join(str(Path.home()), ".shadowai", "api_token")

# Paths that don't require auth
EXEMPT_PATHS = {
    "/api/status",
}


def _load_or_create_token() -> str:
    """Load existing token or generate a new one."""
    try:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, "r") as f:
                token = f.read().strip()
                if token:
                    return token
    except Exception as e:
        log.warning(f"Failed to read token file: {e}")

    # Generate new token
    token = secrets.token_hex(32)
    try:
        os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
        with open(TOKEN_FILE, "w") as f:
            f.write(token)
        log.info(f"API token generated and saved to {TOKEN_FILE}")
    except Exception as e:
        log.error(f"Failed to save token file: {e}")

    return token


# Module-level token loaded once
_api_token = _load_or_create_token()


def get_api_token() -> str:
    """Return the current API token (for daemon or CLI to read)."""
    return _api_token


def _is_localhost(remote_addr: str) -> bool:
    """Check if request comes from localhost."""
    return remote_addr in ("127.0.0.1", "::1", "localhost")


def require_auth():
    """Flask before_request hook that checks bearer token on /api/* routes."""
    # Skip auth in test mode
    if current_app.config.get("TESTING"):
        return None

    # Only protect /api/* routes
    if not request.path.startswith("/api/"):
        return None

    # Exempt specific paths
    if request.path in EXEMPT_PATHS:
        return None

    # Localhost is trusted (daemon, internal processes)
    if _is_localhost(request.remote_addr):
        return None

    # Check Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        if token == _api_token:
            return None

    return jsonify({"error": "Unauthorized"}), 401
