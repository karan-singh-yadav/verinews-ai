"""Rule-based claim and entity extraction.

Honest scope: this is a transparent heuristic (sentence scoring + capitalisation/regex entity
spotting), not a trained NER/claim-detection model. It cannot distinguish people from
organisations from places; it reports capitalised phrases, dates and figures. Replacing it with
spaCy or an LLM later only requires keeping the `Claim` interface.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from by with without
about into over under after before between during is are was were be been being has have had do
does did will would could should may might can cannot not no nor so such as it its he she they
them his her their our your you we i who whom whose which what when where why how also just very
more most said says say according new news report reported reports one two three
""".split())

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\u201c])")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
_CAPWORD = r"(?:[A-Z][a-zA-Z'\-]+|[A-Z]{2,})"
_CAP_PHRASE = re.compile(
    rf"\b{_CAPWORD}(?:\s+(?:(?:of|the|de|van|von|for|and)\s+)?{_CAPWORD})*"
)
_MONTHS = {"January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December", "Jan", "Feb", "Mar", "Apr", "Jun",
           "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec"}
_DATE = re.compile(
    r"\b(?:\d{1,2}\s+)?(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"(?:\s+\d{1,2})?(?:,?\s+\d{4})?\b|\b\d{4}-\d{2}-\d{2}\b|\b(?:19|20)\d{2}\b"
)
_FIGURE = re.compile(
    r"[$\u20b9\u20ac\u00a3]\s?\d[\d,.]*"
    r"|\b\d[\d,.]*\s?(?:%|percent|crore|lakh|million|billion)\b"
    r"|\b\d[\d,]{2,}\b"
)
_CAP_STOP = {"The", "A", "An", "This", "That", "These", "Those", "He", "She", "They", "It", "We",
             "In", "On", "At", "But", "And", "However", "While", "After", "Before", "According",
             "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
_CUES = ("said", "says", "according to", "announced", "confirmed", "claimed", "claims", "reported",
         "killed", "died", "arrested", "banned", "approved", "passed", "resigned", "found",
         "revealed", "admitted", "will", "has been", "have been", "declared", "ordered", "won",
         "lost", "rose", "fell", "increase", "decrease", "launched", "cancelled", "canceled")
_OPINION = ("i think", "i believe", "in my opinion", "should ", "must ", "we need to", "opinion")


@dataclass
class Claim:
    text: str
    entities: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    score: float = 0.0

    def query_text(self, max_len: int = 200) -> str:
        return self.text if len(self.text) <= max_len else self.text[:max_len].rsplit(" ", 1)[0]

    def to_dict(self) -> dict:
        return asdict(self)


def stem(token: str) -> str:
    t = token.lower()
    for suffix in ("ing", "ed", "es", "s"):
        if len(t) > len(suffix) + 3 and t.endswith(suffix):
            return t[: -len(suffix)]
    return t


def content_tokens(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text)
            if len(t) >= 3 and t.lower() not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def extract_entities(sentence: str) -> list[str]:
    """Capitalised phrases (people/orgs/places are not distinguished). A lone capitalised word
    at the start of the sentence is ignored because it is usually just ordinary capitalisation."""
    ents: list[str] = []
    for m in _CAP_PHRASE.finditer(sentence):
        words = m.group(0).split()
        while words and (words[0] in _CAP_STOP or words[0] in _MONTHS):
            words.pop(0)
        while words and words[-1] in _MONTHS:
            words.pop()
        if not words:
            continue
        phrase = " ".join(words)
        if m.start() == 0 and len(words) == 1 and not phrase.isupper():
            continue
        if len(phrase) >= 2 and phrase not in ents:
            ents.append(phrase)
    return ents[:6]


def _score(sentence: str, entities: list[str], figures: list[str]) -> float:
    low = sentence.lower()
    n_words = len(sentence.split())
    if n_words < 6 or n_words > 60 or sentence.rstrip().endswith("?"):
        return -1.0
    s = 2.0 if figures else 0.0
    s += 1.5 * min(2, sum(1 for c in _CUES if c in low))
    s += min(3, len(entities))
    s += 1.0 if ('"' in sentence or "\u201c" in sentence) else 0.0
    s -= 2.0 if any(o in low for o in _OPINION) else 0.0
    return s


def extract_claims(text: str, max_claims: int = 3) -> list[Claim]:
    """Pick the most check-worthy sentences. The first sentence (usually headline + lede)
    gets a small bonus because it normally states the main claim."""
    sentences = split_sentences(text)
    claims: list[Claim] = []
    for i, sent in enumerate(sentences[:40]):
        ents = extract_entities(sent)
        figures = [m.group(0).strip() for m in _FIGURE.finditer(sent)]
        score = _score(sent, ents, figures)
        if score < 0:
            continue
        if i == 0:
            score += 1.0
        kw = list(dict.fromkeys(content_tokens(sent)))
        claims.append(Claim(
            text=sent[:400], entities=ents,
            dates=[m.group(0).strip() for m in _DATE.finditer(sent)][:4],
            figures=figures[:4], keywords=kw[:15], score=score,
        ))
    claims.sort(key=lambda c: -c.score)
    chosen = [c for c in claims if c.score > 0][:max_claims]
    if not chosen and sentences:  # short input: treat the whole text as one claim
        whole = " ".join(sentences)[:400]
        chosen = [Claim(text=whole, entities=extract_entities(whole),
                        keywords=list(dict.fromkeys(content_tokens(whole)))[:15], score=0.0)]
    return chosen
