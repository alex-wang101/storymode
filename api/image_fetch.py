"""Fetch a reference image from a user-supplied URL into a temp file.

Threat model: the URL is attacker-controlled. The API must not be turned
into a reflector that connects to internal services on the operator's
behalf. The guards below enforce:

  - https only (blocks ``file://``, ``ftp://``, ``gopher://``, and also
    plain ``http://`` so cloud metadata endpoints like
    ``http://169.254.169.254`` can't be reached even by IP).
  - DNS pre-flight: every A/AAAA record the hostname resolves to must be
    a public address. Private, loopback, link-local, multicast, reserved,
    and IPv4-mapped IPv6 addresses are all rejected.
  - Redirects re-validated at each hop, hop count capped.
  - Streaming download with a hard byte ceiling so a 10 GB jpeg can't
    drain disk.
  - Content-Type must start with ``image/``.

Known limitation: there is a TOCTOU / DNS-rebinding window between the
pre-flight resolution and the actual TCP connect. Closing it fully
requires forcing httpx to dial the pre-validated IP and override SNI;
out of scope for a localhost dev tool. Revisit before exposing publicly.
"""

from __future__ import annotations

import ipaddress
import mimetypes
import os
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx


_ALLOWED_SCHEMES = frozenset({"https"})
_MAX_REDIRECTS = 2
_TIMEOUT_SECONDS = 10.0


class ImageFetchError(Exception):
    """Any URL-validation or download failure. Message is safe to surface in 4xx body."""


def _is_public_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    # IPv4-mapped IPv6 (::ffff:10.0.0.1) must be checked as its IPv4 form.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _validate_url_target(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ImageFetchError(
            f"scheme {parsed.scheme!r} not allowed; must be one of {sorted(_ALLOWED_SCHEMES)}"
        )
    if not parsed.hostname:
        raise ImageFetchError("URL is missing a hostname")
    port = parsed.port or 443
    try:
        addrinfo = socket.getaddrinfo(parsed.hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ImageFetchError(f"DNS resolution failed for {parsed.hostname}: {e}") from None
    if not addrinfo:
        raise ImageFetchError(f"DNS returned no addresses for {parsed.hostname}")
    for _, _, _, _, sockaddr in addrinfo:
        ip_str = sockaddr[0]
        if not _is_public_ip(ip_str):
            raise ImageFetchError(
                f"hostname {parsed.hostname} resolved to non-public IP {ip_str}; refusing"
            )


def _extension_for(content_type: str) -> str:
    base = content_type.split(";", 1)[0].strip().lower()
    # mimetypes prefers .jpe over .jpg for image/jpeg on some platforms; normalise.
    if base in ("image/jpeg", "image/jpg"):
        return ".jpg"
    return mimetypes.guess_extension(base) or ".bin"


def fetch_image_to_temp(url: str, *, max_bytes: int) -> Path:
    """Download ``url`` to a fresh temp file. Raises ImageFetchError on any failure."""
    current_url = url
    resp: httpx.Response | None = None

    with httpx.Client(timeout=_TIMEOUT_SECONDS, follow_redirects=False) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            _validate_url_target(current_url)
            req = client.build_request("GET", current_url)
            r = client.send(req, stream=True)
            if r.is_redirect:
                next_url = r.headers.get("location")
                r.close()
                if not next_url:
                    raise ImageFetchError(
                        f"redirect from {current_url} missing Location header"
                    )
                current_url = str(httpx.URL(current_url).join(next_url))
                continue
            resp = r
            break
        else:
            raise ImageFetchError(f"exceeded {_MAX_REDIRECTS} redirects")

        assert resp is not None
        try:
            if resp.status_code != 200:
                raise ImageFetchError(f"HTTP {resp.status_code} from {current_url}")
            content_type = resp.headers.get("content-type", "")
            if not content_type.lower().startswith("image/"):
                raise ImageFetchError(f"content-type {content_type!r} is not image/*")
            declared = resp.headers.get("content-length")
            if declared is not None:
                try:
                    if int(declared) > max_bytes:
                        raise ImageFetchError(
                            f"declared content-length {declared} exceeds {max_bytes}"
                        )
                except ValueError:
                    pass  # malformed header; the streaming cap below still applies

            fd, tmp_name = tempfile.mkstemp(
                prefix="videofind_ref_", suffix=_extension_for(content_type)
            )
            dest = Path(tmp_name)
            total = 0
            try:
                with os.fdopen(fd, "wb") as f:
                    for chunk in resp.iter_bytes():
                        total += len(chunk)
                        if total > max_bytes:
                            raise ImageFetchError(
                                f"image body exceeded {max_bytes} bytes"
                            )
                        f.write(chunk)
            except BaseException:
                dest.unlink(missing_ok=True)
                raise
            return dest
        finally:
            resp.close()
