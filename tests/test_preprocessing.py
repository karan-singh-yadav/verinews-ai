import math

from ml.preprocessing import clean_text, strip_source_artifacts


def test_clean_text_basic_normalisation():
    out = clean_text("  Hello <b>WORLD</b>!  Visit https://example.com/x now.  ")
    assert out == "hello world ! visit url now."


def test_clean_text_handles_none_nan_and_non_str():
    assert clean_text(None) == ""
    assert clean_text(math.nan) == ""
    assert clean_text(12345) == "12345"


def test_reuters_prefix_is_removed_to_prevent_label_leakage():
    raw = "WASHINGTON (Reuters) - The senate passed the bill on Tuesday."
    assert "reuters" not in clean_text(raw).lower()
    assert "senate passed the bill" in clean_text(raw)


def test_source_leak_removal_can_be_disabled():
    assert "reuters" in clean_text("LONDON (Reuters) - Hello", remove_source_leaks=False)


def test_strip_known_boilerplate():
    assert "21st century wire" not in strip_source_artifacts("21st Century Wire says nothing").lower()
    assert "pic.twitter.com" not in strip_source_artifacts("look pic.twitter.com/abc123")


def test_email_replaced():
    assert clean_text("contact me at a.b@example.com") == "contact me at email"
