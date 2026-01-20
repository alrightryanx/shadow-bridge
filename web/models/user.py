"""
User Model - Multi-user support for ShadowAI collaboration
"""

import hashlib
import os
import secrets
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class UserDevice:
    """Device linked to a user account."""
    device_id: str
    device_name: str
    platform: str  # "android", "web", "desktop"
    linked_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_seen: Optional[str] = None
    push_token: Optional[str] = None  # For push notifications

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserDevice":
        return cls(**data)


@dataclass
class User:
    """User account for ShadowAI collaboration."""
    id: str
    email: str
    password_hash: str
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_login: Optional[str] = None
    is_active: bool = True
    devices: List[UserDevice] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)

    # Collaboration preferences
    presence_visible: bool = True  # Show online status to others
    allow_invites: bool = True  # Allow others to invite to shared sessions

    @staticmethod
    def generate_id() -> str:
        """Generate a unique user ID."""
        return f"user_{secrets.token_hex(12)}"

    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> str:
        """
        Hash password using PBKDF2-HMAC-SHA256.
        Returns: salt:hash as hex string
        """
        if salt is None:
            salt = os.urandom(32)

        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000,  # iterations
            dklen=32
        )

        return f"{salt.hex()}:{key.hex()}"

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify password against stored hash."""
        try:
            salt_hex, hash_hex = password_hash.split(':')
            salt = bytes.fromhex(salt_hex)
            expected_hash = bytes.fromhex(hash_hex)

            key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                100000,
                dklen=32
            )

            # Constant-time comparison to prevent timing attacks
            return secrets.compare_digest(key, expected_hash)
        except (ValueError, AttributeError):
            return False

    def check_password(self, password: str) -> bool:
        """Check if password matches this user's hash."""
        return self.verify_password(password, self.password_hash)

    def update_password(self, new_password: str):
        """Update user's password."""
        self.password_hash = self.hash_password(new_password)
        self.updated_at = datetime.utcnow().isoformat()

    def add_device(self, device: UserDevice):
        """Link a device to this user."""
        # Remove existing device with same ID if present
        self.devices = [d for d in self.devices if d.device_id != device.device_id]
        self.devices.append(device)
        self.updated_at = datetime.utcnow().isoformat()

    def remove_device(self, device_id: str):
        """Unlink a device from this user."""
        self.devices = [d for d in self.devices if d.device_id != device_id]
        self.updated_at = datetime.utcnow().isoformat()

    def get_device(self, device_id: str) -> Optional[UserDevice]:
        """Get a specific linked device."""
        for device in self.devices:
            if device.device_id == device_id:
                return device
        return None

    def update_device_last_seen(self, device_id: str):
        """Update device's last seen timestamp."""
        device = self.get_device(device_id)
        if device:
            device.last_seen = datetime.utcnow().isoformat()

    def to_dict(self, include_password: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = {
            "id": self.id,
            "email": self.email,
            "display_name": self.display_name,
            "avatar_url": self.avatar_url,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_login": self.last_login,
            "is_active": self.is_active,
            "devices": [d.to_dict() for d in self.devices],
            "settings": self.settings,
            "presence_visible": self.presence_visible,
            "allow_invites": self.allow_invites,
        }
        if include_password:
            data["password_hash"] = self.password_hash
        return data

    def to_public_dict(self) -> Dict[str, Any]:
        """Convert to dictionary safe for public/API responses."""
        return {
            "id": self.id,
            "email": self.email,
            "display_name": self.display_name or self.email.split('@')[0],
            "avatar_url": self.avatar_url,
            "presence_visible": self.presence_visible,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        """Create User from dictionary."""
        devices_data = data.pop("devices", [])
        devices = [UserDevice.from_dict(d) for d in devices_data]
        return cls(**data, devices=devices)

    @classmethod
    def create(cls, email: str, password: str, display_name: Optional[str] = None) -> "User":
        """Create a new user with hashed password."""
        return cls(
            id=cls.generate_id(),
            email=email.lower().strip(),
            password_hash=cls.hash_password(password),
            display_name=display_name,
        )


@dataclass
class Session:
    """User session for authentication."""
    id: str
    user_id: str
    device_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    @staticmethod
    def generate_id() -> str:
        """Generate a secure session token."""
        return secrets.token_urlsafe(32)

    def is_expired(self) -> bool:
        """Check if session has expired."""
        if not self.expires_at:
            return False
        return datetime.fromisoformat(self.expires_at) < datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(**data)

    @classmethod
    def create(cls, user_id: str, device_id: Optional[str] = None,
               expires_hours: int = 24 * 7) -> "Session":
        """Create a new session with expiration."""
        from datetime import timedelta
        expires_at = (datetime.utcnow() + timedelta(hours=expires_hours)).isoformat()
        return cls(
            id=cls.generate_id(),
            user_id=user_id,
            device_id=device_id,
            expires_at=expires_at,
        )
