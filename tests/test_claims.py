from backend.app.services.claims import (
    extract_claims,
    extract_entities,
    split_sentences,
    stem,
)


def test_entities_merge_multiword_names_and_skip_months_and_sentence_initial_words():
    ents = extract_entities("Prime Minister Narendra Modi met officials of the World Health "
                            "Organization in New Delhi on 3 May 2026.")
    assert "Prime Minister Narendra Modi" in ents and "World Health Organization" in ents
    assert "New Delhi" in ents and not any("May" in e for e in ents)
    assert extract_entities("Officials said the plan costs money.") == []


def test_check_worthy_sentences_rank_above_questions_and_opinion():
    text = ("The health ministry announced on 12 March 2026 that 4,500 new clinics will open in "
            "Uttar Pradesh. What do you think about this plan? I think we should all be happy "
            "about the plan overall.")
    claims = extract_claims(text, max_claims=3)
    assert len(claims) == 1 and "4,500" in claims[0].text
    assert claims[0].dates and claims[0].figures and "Uttar Pradesh" in claims[0].entities


def test_short_text_falls_back_to_single_claim():
    claims = extract_claims("Vaccine causes outbreak", max_claims=2)
    assert len(claims) == 1 and claims[0].keywords


def test_empty_text_yields_no_claims():
    assert extract_claims("   ") == []


def test_max_claims_respected_and_keywords_exclude_stopwords():
    text = " ".join(f"The agency announced that {n} thousand people were arrested in Mumbai "
                    f"during protest number {n}." for n in range(1, 8))
    claims = extract_claims(text, max_claims=2)
    assert len(claims) == 2
    assert "the" not in claims[0].keywords and "that" not in claims[0].keywords


def test_sentence_split_and_stem():
    assert len(split_sentences("One thing. Another thing! Third?")) == 3
    assert stem("arrested") == stem("arrests")
