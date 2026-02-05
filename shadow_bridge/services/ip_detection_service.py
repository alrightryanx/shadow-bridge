"""
IP Detection Service for ShadowBridge

Provides comprehensive IP detection and classification for network awareness.
Supports local, VPN (Tailscale), and external IP detection.
"""

import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from collections import defaultdict
import ipaddress


@dataclass
class IPInfo:
    """Information about a detected IP address"""

    address: str
    type: str  # local, tailscale, vpn, external
    interface: Optional[str] = None
    is_primary: bool = False
    reachable: bool = True


class IPDetectionService:
    """
    Service for detecting and classifying IP addresses on the system.

    Features:
    - Detects all network interfaces and their IPs
    - Classifies IPs as local, Tailscale (100.x.x.x), VPN, or external
    - Caches results to reduce system calls
    - Primary IP detection via connection test
    - Reachability validation
    """

    # IP Range Classifications
    TAILSCALE_PREFIX = "100."
    LOCAL_NETWORKS = [
        ipaddress.IPv4Network("192.168.0.0/16"),
        ipaddress.IPv4Network("10.0.0.0/8"),
        ipaddress.IPv4Network("172.16.0.0/12"),
    ]
    LOOPBACK_PREFIX = "127."

    # Cache settings
    CACHE_TTL_SECONDS = 30
    EXTERNAL_CACHE_TTL_SECONDS = 300  # 5 minutes for external IPs

    def __init__(self):
        self._lock = threading.Lock()
        self._cache: Dict[str, IPInfo] = {}
        self._cache_time: float = 0
        self._primary_ip_cache: Optional[str] = None
        self._primary_ip_cache_time: float = 0
        self._external_ip_cache: Optional[str] = None
        self._external_ip_cache_time: float = 0

    def get_all_ips(self, force_refresh: bool = False) -> List[IPInfo]:
        """
        Get all detected IP addresses with classification.

        Args:
            force_refresh: Force refresh of cached data

        Returns:
            List of IPInfo objects
        """
        with self._lock:
            current_time = time.time()
            if (
                not force_refresh
                and (current_time - self._cache_time) < self.CACHE_TTL_SECONDS
            ):
                return list(self._cache.values())

            # Refresh IP detection
            self._detect_ips()
            return list(self._cache.values())

    def get_ips_by_type(
        self, ip_type: str, force_refresh: bool = False
    ) -> List[IPInfo]:
        """
        Get all IPs of a specific type.

        Args:
            ip_type: Type of IP to filter (local, tailscale, vpn, external)
            force_refresh: Force refresh of cached data

        Returns:
            List of IPInfo objects matching the type
        """
        all_ips = self.get_all_ips(force_refresh=force_refresh)
        return [ip for ip in all_ips if ip.type == ip_type]

    def get_primary_ip(self, force_refresh: bool = False) -> Optional[str]:
        """
        Get the primary IP address (the one used for outbound connections).

        Args:
            force_refresh: Force refresh of cached data

        Returns:
            Primary IP address or None if detection fails
        """
        with self._lock:
            current_time = time.time()
            if (
                not force_refresh
                and (current_time - self._primary_ip_cache_time)
                < self.CACHE_TTL_SECONDS
            ):
                return self._primary_ip_cache

            # Detect primary IP via connection test
            try:
                # Create a socket and connect to a reliable external server
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.settimeout(2.0)
                    # Doesn't actually send data, just determines the route
                    s.connect(("8.8.8.8", 80))
                    primary_ip = s.getsockname()[0]
                    self._primary_ip_cache = primary_ip
                    self._primary_ip_cache_time = current_time

                    # Mark as primary in cache
                    if primary_ip in self._cache:
                        self._cache[primary_ip].is_primary = True

                    return primary_ip
            except Exception as e:
                print(f"[IPDetection] Failed to detect primary IP: {e}")
                return None

    def get_external_ip(self, force_refresh: bool = False) -> Optional[str]:
        """
        Get the external/public IP address.

        Args:
            force_refresh: Force refresh of cached data

        Returns:
            External IP address or None if detection fails
        """
        with self._lock:
            current_time = time.time()
            if (
                not force_refresh
                and (current_time - self._external_ip_cache_time)
                < self.EXTERNAL_CACHE_TTL_SECONDS
            ):
                return self._external_ip_cache

            # Try multiple services for external IP detection
            services = [
                ("https://api.ipify.org", "text/plain"),
                ("https://ipinfo.io/ip", "text/plain"),
                ("https://icanhazip.com", "text/plain"),
            ]

            import urllib.request

            for url, content_type in services:
                try:
                    req = urllib.request.Request(url, headers={"Accept": content_type})
                    with urllib.request.urlopen(req, timeout=5) as response:
                        external_ip = response.read().decode().strip()

                        # Validate IP format
                        ipaddress.ip_address(external_ip)  # Will raise if invalid

                        self._external_ip_cache = external_ip
                        self._external_ip_cache_time = current_time
                        return external_ip
                except Exception as e:
                    print(f"[IPDetection] Failed to get external IP from {url}: {e}")
                    continue

            return None

    def get_ip_summary(self, force_refresh: bool = False) -> Dict:
        """
        Get a summary of all detected IPs.

        Args:
            force_refresh: Force refresh of cached data

        Returns:
            Dictionary with categorized IPs and metadata
        """
        all_ips = self.get_all_ips(force_refresh=force_refresh)

        # Categorize IPs
        categories: Dict[str, List[str]] = defaultdict(list)
        for ip_info in all_ips:
            categories[ip_info.type].append(ip_info.address)

        return {
            "local": categories.get("local", []),
            "tailscale": categories.get("tailscale", []),
            "vpn": categories.get("vpn", []),
            "external": [self.get_external_ip(force_refresh)]
            if self.get_external_ip(force_refresh)
            else [],
            "primary": self.get_primary_ip(force_refresh),
            "total_count": len(all_ips),
            "cache_age_seconds": time.time() - self._cache_time,
        }

    def _detect_ips(self):
        """Detect all IPs from network interfaces and classify them."""
        detected_ips = {}

        # Method 1: Get IP addresses from all interfaces via socket.getaddrinfo
        try:
            hostname = socket.gethostname()
            all_addrinfo = socket.getaddrinfo(hostname, None, socket.AF_INET)

            for addrinfo in all_addrinfo:
                ip = str(addrinfo[4][0])
                if ip.startswith(self.LOOPBACK_PREFIX):
                    continue

                ip_info = IPInfo(
                    address=ip,
                    type=self._classify_ip(ip),
                    is_primary=False,
                )
                detected_ips[ip] = ip_info

        except Exception as e:
            print(f"[IPDetection] Error in getaddrinfo: {e}")

        # Method 2: Try to get interface-specific IPs (Linux/Unix)
        try:
            result = subprocess.run(
                ["ip", "addr", "show"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                interface_ips = self._parse_ip_addr_output(result.stdout)
                for ip, interface in interface_ips:
                    if ip in detected_ips:
                        detected_ips[ip].interface = interface
                    else:
                        detected_ips[ip] = IPInfo(
                            address=ip,
                            type=self._classify_ip(ip),
                            interface=interface,
                            is_primary=False,
                        )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"[IPDetection] Could not use 'ip addr' command: {e}")

        # Update cache
        self._cache = detected_ips
        self._cache_time = time.time()

        # Update primary IP flag
        primary_ip = self.get_primary_ip(force_refresh=True)
        if primary_ip and primary_ip in self._cache:
            self._cache[primary_ip].is_primary = True

    def _classify_ip(self, ip: str) -> str:
        """
        Classify an IP address type.

        Args:
            ip: IP address string

        Returns:
            Classification: local, tailscale, vpn, or external
        """
        try:
            ip_obj = ipaddress.IPv4Address(ip)

            # Check for Tailscale
            if ip.startswith(self.TAILSCALE_PREFIX):
                return "tailscale"

            # Check for local/private networks
            if not ip_obj.is_private:
                return "external"

            # Check specific local ranges
            for network in self.LOCAL_NETWORKS:
                if ip_obj in network:
                    return "local"

            # Other private IPs (including VPNs)
            return "vpn"

        except ipaddress.AddressValueError:
            return "unknown"

    def _parse_ip_addr_output(self, output: str) -> List[tuple]:
        """
        Parse 'ip addr show' output to extract interface and IP pairs.

        Args:
            output: Command output string

        Returns:
            List of (ip, interface) tuples
        """
        results = []
        current_interface = None

        for line in output.split("\n"):
            line = line.strip()

            # Detect interface line (e.g., "2: eth0:")
            if ":" in line and not line.startswith(" "):
                parts = line.split(":")
                if len(parts) >= 2:
                    current_interface = parts[1].strip().split("@")[0]

            # Detect inet line (e.g., "    inet 192.168.1.100/24")
            elif "inet " in line and "inet6" not in line:
                parts = line.split()
                if len(parts) >= 2:
                    ip_with_prefix = parts[1]
                    ip = ip_with_prefix.split("/")[0]
                    if current_interface and ip:
                        results.append((ip, current_interface))

        return results

    def clear_cache(self):
        """Clear all cached IP data."""
        with self._lock:
            self._cache.clear()
            self._cache_time = 0
            self._primary_ip_cache = None
            self._primary_ip_cache_time = 0
            self._external_ip_cache = None
            self._external_ip_cache_time = 0
            print("[IPDetection] Cache cleared")

    def is_tailscale_available(self, force_refresh: bool = False) -> bool:
        """
        Check if Tailscale IP is available.

        Args:
            force_refresh: Force refresh of cached data

        Returns:
            True if Tailscale IP detected
        """
        tailscale_ips = self.get_ips_by_type("tailscale", force_refresh=force_refresh)
        return len(tailscale_ips) > 0

    def get_tailscale_ip(self, force_refresh: bool = False) -> Optional[str]:
        """
        Get the first available Tailscale IP.

        Args:
            force_refresh: Force refresh of cached data

        Returns:
            Tailscale IP address or None if not available
        """
        tailscale_ips = self.get_ips_by_type("tailscale", force_refresh=force_refresh)
        return tailscale_ips[0].address if tailscale_ips else None


# Singleton instance
_ip_detection_service = None
_service_lock = threading.Lock()


def get_ip_detection_service() -> IPDetectionService:
    """Get the singleton IPDetectionService instance."""
    global _ip_detection_service
    with _service_lock:
        if _ip_detection_service is None:
            _ip_detection_service = IPDetectionService()
        return _ip_detection_service


# CLI interface for testing
if __name__ == "__main__":
    service = get_ip_detection_service()

    print("=== IP Detection Service ===")
    print("\nAll detected IPs:")
    all_ips = service.get_all_ips()
    for ip_info in all_ips:
        primary_marker = " [PRIMARY]" if ip_info.is_primary else ""
        interface_info = f" ({ip_info.interface})" if ip_info.interface else ""
        print(f"  {ip_info.address} - {ip_info.type}{interface_info}{primary_marker}")

    print("\nIP Summary:")
    summary = service.get_ip_summary()
    for key, value in summary.items():
        if key != "cache_age_seconds":
            print(f"  {key}: {value}")

    print(f"\nCache age: {summary['cache_age_seconds']:.1f}s")

    print("\nTailscale available:", service.is_tailscale_available())
    print("Tailscale IP:", service.get_tailscale_ip())

    print("\nExternal IP:", service.get_external_ip())
