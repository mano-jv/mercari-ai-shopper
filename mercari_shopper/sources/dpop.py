"""Per-request DPoP proofs for api.mercari.jp.

The web client signs an ES256 JWT with a key it generates itself. Nothing is
registered and no account is involved. The public key travels in the JWT
header, and the payload binds the proof to one URL and one HTTP method.
"""

from __future__ import annotations

import time
import uuid
from base64 import urlsafe_b64encode

import jwt
from cryptography.hazmat.primitives.asymmetric import ec


def _b64url_uint(value: int, length: int = 32) -> str:
    raw = value.to_bytes(length, "big")
    return urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


class DpopSigner:
    """One ephemeral P-256 key and a stable device uuid for a client session."""

    def __init__(self) -> None:
        self._key = ec.generate_private_key(ec.SECP256R1())
        self.device_uuid = str(uuid.uuid4())

    def public_jwk(self) -> dict[str, str]:
        numbers = self._key.public_key().public_numbers()
        return {
            "crv": "P-256",
            "kty": "EC",
            "x": _b64url_uint(numbers.x),
            "y": _b64url_uint(numbers.y),
        }

    def proof(self, url: str, method: str, *, now: int | None = None) -> str:
        """Return a DPoP JWT for this exact URL and method."""
        headers = {"typ": "dpop+jwt", "alg": "ES256", "jwk": self.public_jwk()}
        payload = {
            "iat": int(time.time()) if now is None else now,
            "jti": str(uuid.uuid4()),
            "htu": url,
            "htm": method.upper(),
            "uuid": self.device_uuid,
        }
        token = jwt.encode(payload, self._key, algorithm="ES256", headers=headers)
        if isinstance(token, bytes):
            return token.decode("ascii")
        return token
