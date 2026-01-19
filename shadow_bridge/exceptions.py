"""
ShadowBridge Custom Exceptions
------------------------------
Structured exception hierarchy for better error handling and debugging.
"""


class ShadowBridgeError(Exception):
    """Base exception for all ShadowBridge errors."""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> dict:
        """Serialize exception for logging/JSON responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "code": self.error_code,
            "details": self.details,
        }


# ============ Network Errors ============

class NetworkError(ShadowBridgeError):
    """Base class for network-related errors."""
    pass


class ConnectionError(NetworkError):
    """Failed to establish connection."""
    pass


class TimeoutError(NetworkError):
    """Operation timed out."""
    pass


class PortInUseError(NetworkError):
    """Port is already bound by another process."""
    
    def __init__(self, port: int, message: str = None):
        super().__init__(
            message or f"Port {port} is already in use",
            error_code="PORT_IN_USE",
            details={"port": port}
        )
        self.port = port


class DiscoveryError(NetworkError):
    """Error in network discovery."""
    pass


# ============ SSH Errors ============

class SSHError(ShadowBridgeError):
    """Base class for SSH-related errors."""
    pass


class SSHNotFoundError(SSHError):
    """SSH server not found or not running."""
    
    def __init__(self):
        super().__init__(
            "SSH server not found. Please install and start OpenSSH.",
            error_code="SSH_NOT_FOUND"
        )


class SSHKeyError(SSHError):
    """Error managing SSH keys."""
    pass


class SSHKeyApprovalError(SSHKeyError):
    """Error during SSH key approval process."""
    
    def __init__(self, device_id: str, reason: str):
        super().__init__(
            f"Failed to approve SSH key for device {device_id}: {reason}",
            error_code="SSH_KEY_APPROVAL_FAILED",
            details={"device_id": device_id, "reason": reason}
        )


# ============ Installation Errors ============

class InstallError(ShadowBridgeError):
    """Base class for installation errors."""
    pass


class DependencyError(InstallError):
    """Missing or incompatible dependency."""
    
    def __init__(self, package: str, reason: str = None):
        super().__init__(
            f"Dependency error: {package}" + (f" - {reason}" if reason else ""),
            error_code="DEPENDENCY_ERROR",
            details={"package": package, "reason": reason}
        )
        self.package = package


class PyTorchInstallError(InstallError):
    """Error installing PyTorch."""
    
    def __init__(self, cuda_version: str = None, reason: str = None):
        super().__init__(
            f"PyTorch installation failed" + (f": {reason}" if reason else ""),
            error_code="PYTORCH_INSTALL_FAILED",
            details={"cuda_version": cuda_version, "reason": reason}
        )


# ============ Web Server Errors ============

class WebServerError(ShadowBridgeError):
    """Base class for web server errors."""
    pass


class ServerStartError(WebServerError):
    """Failed to start web server."""
    pass


class ServerShutdownError(WebServerError):
    """Failed to gracefully shutdown web server."""
    pass


# ============ Data Errors ============

class DataError(ShadowBridgeError):
    """Base class for data handling errors."""
    pass


class DataCorruptionError(DataError):
    """Data file is corrupted or invalid."""
    
    def __init__(self, filepath: str, reason: str = None):
        super().__init__(
            f"Data corruption in {filepath}" + (f": {reason}" if reason else ""),
            error_code="DATA_CORRUPTION",
            details={"filepath": filepath, "reason": reason}
        )


class EncryptionError(DataError):
    """Error encrypting or decrypting data."""
    pass


# ============ Singleton Errors ============

class SingleInstanceError(ShadowBridgeError):
    """Another instance is already running."""
    
    def __init__(self):
        super().__init__(
            "Another instance of ShadowBridge is already running",
            error_code="SINGLE_INSTANCE"
        )


class StaleInstanceError(ShadowBridgeError):
    """Stale lock detected from crashed instance."""
    
    def __init__(self):
        super().__init__(
            "Stale instance lock detected - cleaning up",
            error_code="STALE_INSTANCE"
        )
