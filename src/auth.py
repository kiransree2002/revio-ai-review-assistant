"""
Authentication module providing secure password hashing and verification
using PBKDF2-HMAC-SHA256 with cryptographically secure random per-user salts.
"""

import os
import hashlib
import hmac


def hash_password(password: str, salt: bytes = None, iterations: int = 200_000) -> str:
    """
    Hash a plaintext password using PBKDF2-HMAC-SHA256 with a unique salt.
    
    Returns:
        Formatted string: 'pbkdf2:sha256${iterations}${salt_hex}${hash_hex}'
    """
    if not isinstance(password, str) or not password:
        raise ValueError("Password must be a non-empty string.")
    
    if salt is None:
        salt = os.urandom(16)
    
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        iterations
    )
    
    return f"pbkdf2:sha256${iterations}${salt.hex()}${key.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """
    Verify a plaintext password against a stored PBKDF2-HMAC-SHA256 hash string.
    Uses hmac.compare_digest for constant-time comparison against timing attacks.
    """
    if not password or not stored_hash:
        return False
    
    try:
        algorithm, iterations_str, salt_hex, key_hex = stored_hash.split('$')
        if algorithm != 'pbkdf2:sha256':
            return False
        
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        
        computed_key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            iterations
        )
        
        return hmac.compare_digest(computed_key, expected_key)
    except Exception:
        return False
