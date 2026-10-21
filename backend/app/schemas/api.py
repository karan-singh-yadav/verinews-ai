from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_TEXT_CHARS = 20000
DISCLAIMER = (
    "The ML result estimates writing style and vocabulary patterns learned from one dataset. "
    "It is not proof that a claim is true or false. Online evidence is shown separately; "
    "review the cited sources yourself."
)


def _strip_nonempty(v: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError("text must not be empty")
    return v


class PredictRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_CHARS, description="Article text to classify")
    _v = field_validator("text")(_strip_nonempty)


class VerifyRequest(PredictRequest):
    force_verification: bool = Field(
        False, description="Verify online even if the model is confident and the text is familiar")


class VerifyUrlRequest(BaseModel):
    url: str = Field(..., min_length=8, max_length=2048)
    force_verification: bool = False


class FeedbackRequest(BaseModel):
    request_id: str | None = Field(None, max_length=64)
    user_label: Literal["fake", "real", "unsure"]
    predicted_label: Literal["fake", "real"] | None = None
    model_version: str | None = Field(None, max_length=64)
    comment: str | None = Field(None, max_length=500)
    consent_store_text: bool = Field(
        False, description="If true, the article text is stored with the feedback")
    text: str | None = Field(None, max_length=MAX_TEXT_CHARS)


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str | None = None
