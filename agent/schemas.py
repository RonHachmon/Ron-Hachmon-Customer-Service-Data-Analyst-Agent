"""Pydantic input schemas for the agent's tools.

Each tool gets its own schema so:

1. The LLM sees per-field descriptions verbatim in the OpenAI tool spec.
2. We can validate ``category`` / ``intent`` against the loaded dataset and
   raise a ``ValueError`` whose message names the valid options. When that
   error propagates back through the ReAct loop, the model gets a chance to
   correct itself on the next step — which is the whole point of the
   self-correction pattern.

Validators normalize input case (category → upper, intent → lower) so the
agent doesn't have to worry about the user's exact phrasing.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from agent.data import get_df


def _normalize_category(value: str | None) -> str | None:
    """Uppercase, strip, and check the value is a known category."""
    if value is None or value == "":
        return None
    valid = set(get_df()["category"].unique())
    normalized = value.strip().upper()
    if normalized not in valid:
        raise ValueError(
            f"Unknown category {value!r}. Valid categories are: {sorted(valid)}"
        )
    return normalized


def _normalize_intent(value: str | None) -> str | None:
    """Lowercase, strip, and check the value is a known intent."""
    if value is None or value == "":
        return None
    valid = set(get_df()["intent"].unique())
    normalized = value.strip().lower()
    if normalized not in valid:
        raise ValueError(
            f"Unknown intent {value!r}. Valid intents are: {sorted(valid)}"
        )
    return normalized


class NoArgs(BaseModel):
    """Empty schema for tools that take no arguments."""


class ListIntentsArgs(BaseModel):
    category: str | None = Field(
        default=None,
        description=(
            "Optional category to filter by, e.g. 'ACCOUNT'. Case-insensitive. "
            "Omit to return every intent in the dataset."
        ),
    )

    @field_validator("category")
    @classmethod
    def _normalize(cls, value: str | None) -> str | None:
        return _normalize_category(value)


class CountRowsArgs(BaseModel):
    category: str | None = Field(
        default=None,
        description="Optional category filter, e.g. 'REFUND'. Case-insensitive.",
    )
    intent: str | None = Field(
        default=None,
        description=(
            "Optional intent filter, e.g. 'get_refund'. Snake_case, "
            "case-insensitive."
        ),
    )

    @field_validator("category")
    @classmethod
    def _norm_category(cls, value: str | None) -> str | None:
        return _normalize_category(value)

    @field_validator("intent")
    @classmethod
    def _norm_intent(cls, value: str | None) -> str | None:
        return _normalize_intent(value)


class IntentDistributionArgs(BaseModel):
    category: str = Field(
        description=(
            "Category to compute the intent distribution for, e.g. 'ACCOUNT'. "
            "Required."
        ),
    )

    @field_validator("category")
    @classmethod
    def _norm(cls, value: str) -> str:
        normalized = _normalize_category(value)
        if normalized is None:
            raise ValueError("category is required and cannot be empty")
        return normalized


class GetExamplesArgs(BaseModel):
    category: str | None = Field(
        default=None,
        description="Optional category filter, e.g. 'SHIPPING'.",
    )
    intent: str | None = Field(
        default=None,
        description="Optional intent filter, e.g. 'change_shipping_address'.",
    )
    n: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of example rows to return. Default 3, max 20.",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of matching rows to skip before returning. Use this for "
            "'show me N more' follow-ups: to get the next 3 after showing 3, "
            "pass n=3, offset=3."
        ),
    )

    @field_validator("category")
    @classmethod
    def _norm_category(cls, value: str | None) -> str | None:
        return _normalize_category(value)

    @field_validator("intent")
    @classmethod
    def _norm_intent(cls, value: str | None) -> str | None:
        return _normalize_intent(value)


class SearchInstructionsArgs(BaseModel):
    query: str = Field(
        min_length=1,
        description=(
            "Keyword or short phrase to search for inside the user-instruction "
            "column. Case-insensitive substring match — pick short, content-"
            "rich terms ('money back', 'wrong address', 'cancel')."
        ),
    )
    n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of matches to return. Default 5, max 20.",
    )
