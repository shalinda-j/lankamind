"""
network/discovery.py
---------------------
mDNS / Zeroconf LAN auto-discovery for LankaMind.

Any device on the same Wi-Fi network can find the LankaMind API server
automatically — no IP addresses, no manual configuration.

How it works
------------
1. The machine running `lankamind serve` calls `announce()`.
   It broadcasts `_lankamind._tcp.local.` via mDNS so every device on the
   LAN knows the service is available.

2. Any device (phone, tablet, laptop) can:
   a. Open a browser and go to  http://lankamind.local:8000   (mDNS name)
   b. Or use `LankaMindBrowser` to discover the IP+port programmatically.

3. Workers also announce themselves so the gateway can find them without
   any --gateway flag.

Service naming
--------------
Service type : _lankamind._tcp.local.
Instance     : "LankaMind API"  /  "LankaMind Worker 0"  etc.

Fallback
--------
If zeroconf is unavailable or mDNS is blocked, `announce()` logs a warning
and returns None — the rest of the system continues to work with explicit IPs.

Usage
-----
    from network.discovery import announce, browse_once, LankaMindBrowser

    # On the server:
    zc, info = announce(service_name="LankaMind API", port=8000, role="api")

    # On a client / another worker:
    services = browse_once(timeout=3.0)
    for svc in services:
        print(svc)   # {'name': ..., 'host': ..., 'port': ..., 'role': ...}

    # Cleanup:
    zc.unregister_service(info)
    zc.close()
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

SERVICE_TYPE = "_lankamind._tcp.local."


def _local_ip() -> str:
    """Best-effort: return the machine's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def announce(
    service_name: str = "LankaMind API",
    port: int = 8000,
    role: str = "api",
    extra_props: Optional[Dict[str, str]] = None,
    host_ip: Optional[str] = None,
) -> Tuple[object, object] | Tuple[None, None]:
    """
    Announce a LankaMind service on the local network via mDNS.

    Parameters
    ----------
    service_name : Human-readable name (e.g. "LankaMind API", "Worker 0").
    port         : TCP port the service listens on.
    role         : One of "api", "gateway", "worker", "bootstrap".
    extra_props  : Extra key→value properties to broadcast (e.g. model name).
    host_ip      : Override the LAN IP (auto-detected if None).

    Returns
    -------
    (Zeroconf instance, ServiceInfo) on success, (None, None) on failure.
    """
    try:
        from zeroconf import ServiceInfo, Zeroconf
    except ImportError:
        log.warning(
            "zeroconf not installed — mDNS auto-discovery disabled. "
            "Install with: pip install zeroconf"
        )
        return None, None

    ip = host_ip or _local_ip()
    props: Dict[bytes, bytes] = {
        b"role":    role.encode(),
        b"version": b"0.5.0",
    }
    if extra_props:
        for k, v in extra_props.items():
            props[k.encode()] = v.encode()

    # Sanitise service name for DNS: replace spaces → dashes
    dns_name = service_name.replace(" ", "-").lower()
    full_name = f"{service_name}.{SERVICE_TYPE}"

    info = ServiceInfo(
        SERVICE_TYPE,
        full_name,
        addresses=[socket.inet_aton(ip)],
        port=port,
        properties=props,
        server=f"{dns_name}.local.",
    )

    try:
        zc = Zeroconf()
        zc.register_service(info)
        log.info(
            "mDNS: announced '%s' → %s:%d  (http://%s.local:%d)",
            service_name, ip, port, dns_name, port,
        )
        return zc, info
    except Exception as exc:
        log.warning("mDNS announce failed: %s", exc)
        return None, None


def browse_once(
    timeout: float = 3.0,
    service_type: str = SERVICE_TYPE,
) -> List[Dict[str, str]]:
    """
    Synchronously browse for LankaMind services on the LAN.

    Blocks for *timeout* seconds then returns whatever was found.

    Returns
    -------
    List of dicts: [{"name": str, "host": str, "port": int, "role": str}, ...]
    """
    try:
        from zeroconf import ServiceBrowser, Zeroconf
    except ImportError:
        log.warning("zeroconf not installed — cannot browse for services")
        return []

    found: List[Dict[str, str]] = []
    lock = threading.Lock()

    class _Handler:
        def add_service(self, zc: "Zeroconf", stype: str, name: str) -> None:
            info = zc.get_service_info(stype, name, timeout=1000)
            if info:
                role = info.properties.get(b"role", b"unknown").decode()
                addrs = info.parsed_scoped_addresses()
                host = addrs[0] if addrs else "?"
                with lock:
                    found.append({
                        "name": info.name,
                        "host": host,
                        "port": info.port,
                        "role": role,
                    })

        def remove_service(self, *_: object) -> None:
            pass

        def update_service(self, *_: object) -> None:
            pass

    zc = Zeroconf()
    handler = _Handler()
    browser = ServiceBrowser(zc, service_type, handler)
    time.sleep(timeout)
    zc.close()
    return found


class LankaMindBrowser:
    """
    Continuously monitor the LAN for LankaMind services.

    Usage
    -----
        browser = LankaMindBrowser(on_found=lambda svc: print("Found:", svc))
        browser.start()
        # ... later ...
        browser.stop()
        services = browser.services
    """

    def __init__(
        self,
        on_found: Optional[Callable[[Dict[str, str]], None]] = None,
        on_removed: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._on_found = on_found
        self._on_removed = on_removed
        self._lock = threading.Lock()
        self._services: Dict[str, Dict[str, str]] = {}
        self._zc: object | None = None
        self._browser: object | None = None

    @property
    def services(self) -> List[Dict[str, str]]:
        with self._lock:
            return list(self._services.values())

    def start(self) -> None:
        try:
            from zeroconf import ServiceBrowser, Zeroconf
        except ImportError:
            log.warning("zeroconf not installed — LAN browser disabled")
            return

        parent = self

        class _Handler:
            def add_service(self, zc: "Zeroconf", stype: str, name: str) -> None:
                info = zc.get_service_info(stype, name, timeout=1000)
                if not info:
                    return
                role = info.properties.get(b"role", b"unknown").decode()
                addrs = info.parsed_scoped_addresses()
                host = addrs[0] if addrs else "?"
                svc = {"name": name, "host": host, "port": info.port, "role": role}
                with parent._lock:
                    parent._services[name] = svc
                if parent._on_found:
                    parent._on_found(svc)

            def remove_service(self, _: object, __: str, name: str) -> None:
                with parent._lock:
                    parent._services.pop(name, None)
                if parent._on_removed:
                    parent._on_removed(name)

            def update_service(self, *_: object) -> None:
                pass

        self._zc = Zeroconf()
        self._browser = ServiceBrowser(self._zc, SERVICE_TYPE, _Handler())

    def stop(self) -> None:
        if self._zc:
            self._zc.close()
            self._zc = None
