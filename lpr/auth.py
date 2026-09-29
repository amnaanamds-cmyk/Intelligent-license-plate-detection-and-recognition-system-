"""User accounts, roles and login sessions.

Roles (each includes everything the roles before it can do):
    viewer    see the dashboard, events and evidence images, search, export
    operator  also scan images, manage the watchlist, start and stop cameras
    admin     also manage users and read the audit log

Passwords are stored as salted PBKDF2-SHA256 hashes. A login returns a random
bearer token that is valid for ``ttl_hours``. Tokens live in memory, so a
restart logs everyone out. The ``api.api_key`` in the config also works as an
admin credential for machine-to-machine integrations.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass

ROLES = ("viewer", "operator", "admin")
_ITERATIONS = 200_000


def hash_password(password: str, iterations: int = _ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex),
                                 int(iters))
    return hmac.compare_digest(digest.hex(), digest_hex)


def role_at_least(role: str, required: str) -> bool:
    return ROLES.index(role) >= ROLES.index(required)


@dataclass
class Principal:
    username: str
    role: str


class AuthManager:
    def __init__(self, store, api_key: str | None = None, ttl_hours: float = 12.0,
                 min_password_length: int = 8):
        self.store = store
        self.api_key = api_key or None
        self.ttl = ttl_hours * 3600
        self.min_password_length = min_password_length
        self._sessions: dict[str, tuple[Principal, float]] = {}
        self._lock = threading.Lock()
        self._failures: dict[str, list[float]] = {}

    @property
    def enabled(self) -> bool:
        """Authentication is enforced once an API key or any user exists.
        With neither (a fresh install) the system runs in open setup mode
        and the web app asks for the first admin account to be created.
        Once a user has ever existed the system never returns to setup mode,
        even if every user is deleted, so it cannot be reopened by accident."""
        return (bool(self.api_key) or self.store.count_users() > 0
                or self.store.get_setting("setup_complete") == "1")

    # ------------------------------------------------------------- users --
    def create_user(self, username: str, password: str, role: str = "operator") -> None:
        username = username.strip()
        if not username or len(username) > 64 or not username.replace("_", "").replace(
                ".", "").replace("-", "").isalnum():
            raise ValueError("username must be 1-64 letters, digits, '.', '_' or '-'")
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        if len(password) < self.min_password_length:
            raise ValueError(f"password must be at least {self.min_password_length} characters")
        self.store.add_user(username, hash_password(password), role)
        self.store.set_setting("setup_complete", "1")

    def delete_user(self, username: str) -> bool:
        user = self.store.get_user(username)
        if user and user["role"] == "admin" and self.store.count_users("admin") <= 1:
            raise ValueError("cannot delete the last admin account")
        with self._lock:
            for tok, (p, _) in list(self._sessions.items()):
                if p.username == username:
                    del self._sessions[tok]
        return self.store.delete_user(username)

    # ---------------------------------------------------------- sessions --
    def _rate_limited(self, username: str) -> bool:
        """At most 5 failed logins per user in 5 minutes (slows password guessing)."""
        now = time.time()
        recent = [t for t in self._failures.get(username, []) if now - t < 300]
        self._failures[username] = recent
        return len(recent) >= 5

    def login(self, username: str, password: str) -> tuple[str, Principal] | None:
        with self._lock:
            if self._rate_limited(username):
                raise PermissionError("too many failed attempts, try again in 5 minutes")
        user = self.store.get_user(username)
        if user is None or not verify_password(password, user["pw_hash"]):
            with self._lock:
                self._failures.setdefault(username, []).append(time.time())
            return None
        token = secrets.token_urlsafe(32)
        principal = Principal(user["username"], user["role"])
        with self._lock:
            self._failures.pop(username, None)
            self._sessions[token] = (principal, time.time() + self.ttl)
        return token, principal

    def logout(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)

    def authenticate(self, api_key: str | None = None,
                     token: str | None = None) -> Principal | None:
        if not self.enabled:
            return Principal("setup", "admin")
        if api_key and self.api_key and hmac.compare_digest(api_key, str(self.api_key)):
            return Principal("api-key", "admin")
        if token:
            with self._lock:
                entry = self._sessions.get(token)
                if entry is None:
                    return None
                principal, expires = entry
                if time.time() > expires:
                    del self._sessions[token]
                    return None
            # A deleted user or a changed role takes effect immediately.
            user = self.store.get_user(principal.username)
            if user is None:
                self.logout(token)
                return None
            return Principal(user["username"], user["role"])
        return None
