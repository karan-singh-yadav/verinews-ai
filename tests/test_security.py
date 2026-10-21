import asyncio

import httpx
import pytest

from backend.app.core.security import (
    FetchError,
    UnsafeURLError,
    UnsupportedContentError,
    fetch_url,
    is_public_ip,
    validate_url,
)

run = asyncio.run
PUBLIC = "93.184.216.34"


def resolver_for(mapping):
    async def _r(host, port):
        return mapping.get(host, [PUBLIC])
    return _r


@pytest.mark.parametrize("url", [
    "http://localhost/", "http://127.0.0.1/", "http://127.1.2.3:80/", "http://10.0.0.5/",
    "http://172.16.0.1/", "http://192.168.1.1/", "http://169.254.169.254/latest/meta-data/",
    "http://100.64.0.1/", "http://0.0.0.0/", "http://[::1]/", "http://[::ffff:127.0.0.1]/",
    "http://[fe80::1]/", "http://[fc00::1]/", "http://[64:ff9b::7f00:1]/",
    "http://printer.local/", "http://db.internal/",
])
def test_private_and_local_targets_are_blocked(url):
    with pytest.raises(UnsafeURLError):
        run(validate_url(url, resolver_for({})))


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com/x", "gopher://example.com", "javascript:alert(1)",
    "http://user:pass@example.com/", "http://example.com:8080/", "http://example.com:22/",
    "", "   ", "http://", "http://exa mple.com/", "http://example.com/\r\nX-Evil: 1",
    "http://" + "a" * 3000 + ".com",
])
def test_malformed_or_disallowed_urls_are_blocked(url):
    with pytest.raises(UnsafeURLError):
        run(validate_url(url, resolver_for({})))


def test_decimal_and_octal_ip_tricks_blocked_via_resolution():
    # urlsplit keeps "2130706433" as a hostname; the OS resolves it to 127.0.0.1
    with pytest.raises(UnsafeURLError):
        run(validate_url("http://2130706433/", resolver_for({"2130706433": ["127.0.0.1"]})))


def test_hostname_resolving_to_private_ip_is_blocked():
    with pytest.raises(UnsafeURLError):
        run(validate_url("http://evil.example/", resolver_for({"evil.example": ["10.1.2.3"]})))


def test_mixed_public_and_private_answers_are_blocked():
    with pytest.raises(UnsafeURLError):
        run(validate_url("http://rebind.example/",
                         resolver_for({"rebind.example": [PUBLIC, "127.0.0.1"]})))


def test_public_url_is_accepted():
    v = run(validate_url("https://news.example.com/a/b?x=1", resolver_for({})))
    assert v.host == "news.example.com" and v.port == 443 and v.path_query == "/a/b?x=1"
    assert v.ips == [PUBLIC]


def test_is_public_ip_edge_cases():
    import ipaddress
    assert is_public_ip(ipaddress.ip_address("8.8.8.8"))
    assert not is_public_ip(ipaddress.ip_address("::ffff:10.0.0.1"))


HTML_OK = "<html><head><title>T</title></head><body>" + "<p>word " * 5 + "</p></body></html>"


def transport(handler):
    return httpx.MockTransport(handler)


def test_fetch_success_pins_ip_and_keeps_host_header():
    seen = {}

    def handler(req: httpx.Request):
        seen["url_host"], seen["host_header"] = req.url.host, req.headers["host"]
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"},
                              text="<html>hello</html>")
    page = run(fetch_url("https://news.example.com/x", resolver=resolver_for({}),
                         transport=transport(handler)))
    assert seen["url_host"] == PUBLIC          # connection pinned to the validated IP
    assert seen["host_header"] == "news.example.com"
    assert page.status_code == 200 and "hello" in page.text


def test_redirect_to_private_ip_is_blocked():
    def handler(req):
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/"})
    with pytest.raises(UnsafeURLError):
        run(fetch_url("http://a.example/", resolver=resolver_for({}), transport=transport(handler)))


def test_redirect_to_hostname_resolving_private_is_blocked():
    def handler(req):
        return httpx.Response(301, headers={"location": "http://intranet.example/admin"})
    with pytest.raises(UnsafeURLError):
        run(fetch_url("http://a.example/", transport=transport(handler),
                      resolver=resolver_for({"intranet.example": ["192.168.0.9"]})))


def test_redirect_to_other_scheme_is_blocked():
    def handler(req):
        return httpx.Response(302, headers={"location": "file:///etc/passwd"})
    with pytest.raises(UnsafeURLError):
        run(fetch_url("http://a.example/", resolver=resolver_for({}), transport=transport(handler)))


def test_safe_redirect_is_followed():
    def handler(req: httpx.Request):
        if req.headers["host"] == "a.example":
            return httpx.Response(302, headers={"location": "https://b.example/final"})
        return httpx.Response(200, headers={"content-type": "text/html"}, text="ok")
    page = run(fetch_url("http://a.example/", resolver=resolver_for({}),
                         transport=transport(handler)))
    assert page.final_url == "https://b.example/final" and page.redirects == 1


def test_redirect_loop_is_capped():
    def handler(req):
        return httpx.Response(302, headers={"location": "http://a.example/again"})
    with pytest.raises(FetchError, match="Too many redirects"):
        run(fetch_url("http://a.example/", max_redirects=3, resolver=resolver_for({}),
                      transport=transport(handler)))


def test_non_html_content_type_rejected():
    def handler(req):
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF")
    with pytest.raises(UnsupportedContentError):
        run(fetch_url("http://a.example/", resolver=resolver_for({}), transport=transport(handler)))


def test_oversized_response_rejected_by_stream_cap():
    def handler(req):
        return httpx.Response(200, headers={"content-type": "text/html"}, content=b"x" * 5000)
    with pytest.raises(FetchError, match="too large"):
        run(fetch_url("http://a.example/", max_bytes=1000, resolver=resolver_for({}),
                      transport=transport(handler)))


def test_oversized_declared_length_rejected():
    def handler(req):
        return httpx.Response(200, headers={"content-type": "text/html", "content-length": "99999"},
                              content=b"x")
    with pytest.raises(FetchError, match="too large"):
        run(fetch_url("http://a.example/", max_bytes=1000, resolver=resolver_for({}),
                      transport=transport(handler)))


def test_http_error_status_and_timeout_become_fetch_errors():
    with pytest.raises(FetchError, match="HTTP 503"):
        run(fetch_url("http://a.example/", resolver=resolver_for({}),
                      transport=transport(lambda r: httpx.Response(503))))

    def boom(req):
        raise httpx.ConnectTimeout("slow")
    with pytest.raises(FetchError, match="too long"):
        run(fetch_url("http://a.example/", resolver=resolver_for({}), transport=transport(boom)))


def test_unresolvable_host_has_its_own_error_type():
    from backend.app.core.security import HostResolutionError

    async def nothing(host, port):
        return []
    with pytest.raises(HostResolutionError):
        run(validate_url("http://no-such-host.example/", nothing))
