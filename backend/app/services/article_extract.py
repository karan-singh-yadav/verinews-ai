"""Turn fetched HTML into article text. The HTML is UNTRUSTED: we only read text, never
execute or render it, and strip scripts/styles/forms."""
from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

_WS = re.compile(r"\s+")
_DROP = ["script", "style", "noscript", "iframe", "form", "nav", "footer", "aside", "header",
         "svg", "template"]


class ExtractionError(ValueError):
    """Not enough readable article text could be extracted."""


@dataclass
class ExtractedArticle:
    title: str
    text: str
    published: str | None


def extract_article(html: str, max_chars: int = 20000, min_chars: int = 200) -> ExtractedArticle:
    soup = BeautifulSoup(html, "lxml")

    def meta(*names: str) -> str | None:
        for n in names:
            tag = soup.find("meta", attrs={"property": n}) or soup.find("meta", attrs={"name": n})
            if tag and tag.get("content"):
                return _WS.sub(" ", str(tag["content"])).strip()
        return None

    title = meta("og:title", "twitter:title") or (
        _WS.sub(" ", soup.title.get_text()).strip() if soup.title else "")
    published = meta("article:published_time", "og:article:published_time", "date")

    for tag in soup(_DROP):
        tag.decompose()
    container = soup.find("article") or soup.body or soup
    paragraphs = [_WS.sub(" ", p.get_text(" ")).strip() for p in container.find_all("p")]
    body = " ".join(p for p in paragraphs if len(p.split()) >= 4)
    if len(body) < min_chars:  # fall back to all visible text
        body = _WS.sub(" ", container.get_text(" ")).strip()
    if len(body) < min_chars:
        raise ExtractionError(
            "Could not extract enough article text (the page may be paywalled, "
            "JavaScript-rendered, or not an article)."
        )
    return ExtractedArticle(title=title[:300], text=body[:max_chars], published=published)
