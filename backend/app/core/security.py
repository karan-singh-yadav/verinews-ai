"""SSRF-hardened URL validation and fetching for POST /api/verify-url.

Defences:
  * http/https only, no embedded credentials, ports 80/443 only, bounded URL length
  * every hostname is resolved and ALL resulting IPs must be public (blocks loopback, RFC1918,
    link-local incl. cloud metadata 169.254.169.254, CGNAT, reserved, multicast, IPv4-mapped
    / NAT64 / 6to4 IPv6 wrappers of those)
  * redirects are followed manually (max N) and each hop is re-validated
  * the TCP connection is pinned to the validated IP (Host header + TLS SNI keep the real
    hostname) so a DNS answer cannot change between validation and connection (anti-rebinding)
  * response size cap, content-type allow-list, total deadline, no proxy env, no cookies
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

MAX_URL_LENGTH = 2048
ALLOWED_PORTS = {80, 443}
ALLOWED_CONTENT_TYPES = {"text/html", "application/xhtml+xml", "text/plain"}
BLOCKED_HOST_SUFFIXES = (".local", ".localhost", ".internal", ".lan", ".home", ".corp", ".intranet")
USER_AGENT = "VeriNewsBot/0.1 (portfolio fact-check research; contact: see repository)"

_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_6TO4 = ipaddress.ip_network("2002::/16")

Resolver = Callable[[str, int], Awaitable[list[str]]]


class UnsafeURLError(ValueError):
    """The URL is malformed or points somewhere we refuse to connect to."""


class HostResolutionError(UnsafeURLError):
    """The hostname does not resolve (typo, dead domain). Distinct so the API can word it kindly."""


class FetchError(RuntimeError):
    """Network-level failure while retrieving a page (message is safe to show to users)."""


class UnsupportedContentError(FetchError):
    """The URL returned something that is not an HTML/text page."""


@dataclass
class ValidatedURL:
    url: str
    scheme: str
    host: str
    port: int
    path_query: str
    ips: list[str]


@dataclass
class FetchedPage:
    final_url: str
    status_code: int
    content_type: str
    text: str
    redirects: int


def is_public_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return is_public_ip(ip.ipv4_mapped)
        if ip in _NAT64:
            return is_public_ip(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
        if ip in _6TO4:
            return is_public_ip(ipaddress.IPv4Address((int(ip) >> 80) & 0xFFFFFFFF))
    return bool(ip.is_global and not ip.is_multicast)


async def default_resolver(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise HostResolutionError("Could not resolve the hostname; check the URL") from exc
    return list(dict.fromkeys(info[4][0] for info in infos))


async def validate_url(url: str, resolver: Resolver | None = None) -> ValidatedURL:
    if not isinstance(url, str) or not url.strip():
        raise UnsafeURLError("URL is empty")
    url = url.strip()
    if len(url) > MAX_URL_LENGTH:
        raise UnsafeURLError("URL is too long")
    if any(ord(c) < 33 or ord(c) == 127 for c in url):
        raise UnsafeURLError("URL contains whitespace or control characters")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise UnsafeURLError("URL is malformed") from exc
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise UnsafeURLError("Only http and https URLs are allowed")
    if parts.username is not None or parts.password is not None:
        raise UnsafeURLError("URLs with embedded credentials are not allowed")
    host = (parts.hostname or "").rstrip(".").lower()
    if not host:
        raise UnsafeURLError("URL has no hostname")
    port = port or (443 if scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise UnsafeURLError("Only the default web ports (80, 443) are allowed")
    if host == "localhost" or host.endswith(BLOCKED_HOST_SUFFIXES):
        raise UnsafeURLError("Local or internal hostnames are not allowed")

    try:
        literal = ipaddress.ip_address(host)
        candidates = [str(literal)]
    except ValueError:
        try:
            ascii_host = host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise UnsafeURLError("Hostname is not valid") from exc
        candidates = await (resolver or default_resolver)(ascii_host, port)
    if not candidates:
        raise HostResolutionError("Could not resolve the hostname; check the URL")

    ips: list[str] = []
    for cand in candidates:
        try:
            ip = ipaddress.ip_address(cand)
        except ValueError as exc:
            raise UnsafeURLError("Hostname resolved to an invalid address") from exc
        if not is_public_ip(ip):
            raise UnsafeURLError("URL points to a private, local or reserved network address")
        ips.append(str(ip))

    path_query = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    return ValidatedURL(url, scheme, host, port, path_query, ips)


def _decode(body: bytes, content_type_header: str) -> str:
    charset = "utf-8"
    for piece in content_type_header.split(";")[1:]:
        k, _, v = piece.strip().partition("=")
        if k.lower() == "charset" and v:
            charset = v.strip("\"' ")
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


async def _fetch_once(
    start_url: str, *, timeout: float, max_bytes: int, max_redirects: int,
    resolver: Resolver | None, transport: httpx.AsyncBaseTransport | None,
) -> FetchedPage:
    current, redirects = start_url, 0
    async with httpx.AsyncClient(
        transport=transport, timeout=timeout, follow_redirects=False, trust_env=False
    ) as client:
        while True:
            v = await validate_url(current, resolver)
            ip = v.ips[0]
            ip_host = f"[{ip}]" if ":" in ip else ip
            default_port = 443 if v.scheme == "https" else 80
            host_header = v.host if v.port == default_port else f"{v.host}:{v.port}"
            pinned = f"{v.scheme}://{ip_host}:{v.port}{v.path_query}"
            headers = {"Host": host_header, "User-Agent": USER_AGENT,
                       "Accept": "text/html,application/xhtml+xml,text/plain;q=0.8",
                       "Accept-Encoding": "gzip, deflate"}
            ext = {"sni_hostname": v.host} if v.scheme == "https" else {}
            try:
                async with client.stream("GET", pinned, headers=headers, extensions=ext) as resp:
                    if 300 <= resp.status_code < 400:
                        location = resp.headers.get("location")
                        if not location:
                            raise FetchError("The page sent an invalid redirect")
                        redirects += 1
                        if redirects > max_redirects:
                            raise FetchError("Too many redirects")
                        current = urljoin(current, location)
                        continue
                    if resp.status_code != 200:
                        raise FetchError(f"The page returned HTTP {resp.status_code}")
                    ctype_header = resp.headers.get("content-type", "")
                    ctype = ctype_header.split(";")[0].strip().lower()
                    if ctype not in ALLOWED_CONTENT_TYPES:
                        raise UnsupportedContentError(
                            f"Unsupported content type '{ctype or 'unknown'}'; "
                            "only web pages are allowed"
                        )
                    declared = resp.headers.get("content-length", "")
                    if declared.isdigit() and int(declared) > max_bytes:
                        raise FetchError("The page is too large")
                    buf = bytearray()
                    async for chunk in resp.aiter_bytes():
                        buf += chunk
                        if len(buf) > max_bytes:
                            raise FetchError("The page is too large")
                    return FetchedPage(current, resp.status_code, ctype,
                                       _decode(bytes(buf), ctype_header), redirects)
            except httpx.TimeoutException as exc:
                raise FetchError("The page took too long to respond") from exc
            except httpx.HTTPError as exc:
                raise FetchError("The page could not be retrieved") from exc


async def fetch_url(
    url: str, *, timeout: float = 10.0, max_bytes: int = 2_000_000, max_redirects: int = 3,
    resolver: Resolver | None = None, transport: httpx.AsyncBaseTransport | None = None,
) -> FetchedPage:
    """Safely fetch a public web page. Raises UnsafeURLError or FetchError subclasses."""
    try:
        return await asyncio.wait_for(
            _fetch_once(url, timeout=timeout, max_bytes=max_bytes, max_redirects=max_redirects,
                        resolver=resolver, transport=transport),
            timeout=timeout * (max_redirects + 1),
        )
    except TimeoutError as exc:
        raise FetchError("The page took too long to respond") from exc
