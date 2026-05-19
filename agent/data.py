"""Pandas-backed data layer for the Bitext Customer Service dataset.

The dataset is loaded once from Hugging Face (cached on disk after the first
call) and held as a pandas DataFrame for the lifetime of the process. Every
tool in the agent reads from the same DataFrame returned by :func:`get_df` —
there is no other source of truth.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

# Every tool relies on these columns being present.
EXPECTED_COLUMNS = ("flags", "instruction", "category", "intent", "response")


@lru_cache(maxsize=1)
def get_df() -> pd.DataFrame:
    """Return the Bitext dataset as a pandas DataFrame.

    The first call downloads the dataset (via Hugging Face's local cache) and
    converts it to pandas. Subsequent calls return the same in-memory
    DataFrame — memoized for the process lifetime.

    Categories are normalized to uppercase so user queries like ``"SHIPPING"``
    and ``"shipping"`` both match. Intents are kept in their original
    snake_case form (e.g. ``get_refund``) since that is how they appear in
    both the dataset and the assignment prompts.
    """
    logger.info("Loading dataset %s from Hugging Face…", DATASET_ID)
    raw = load_dataset(DATASET_ID, split="train")
    df = raw.to_pandas()

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Bitext dataset is missing expected columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns)}"
        )

    df["category"] = df["category"].str.strip().str.upper()
    df["intent"] = df["intent"].str.strip()

    logger.info("Loaded %d rows from %s", len(df), DATASET_ID)
    return df


def summarize() -> dict[str, object]:
    """Return a small summary of the dataset for startup sanity-checks.

    Cheap — only inspects the cached DataFrame from :func:`get_df`.
    """
    df = get_df()
    return {
        "row_count": int(len(df)),
        "categories": sorted(df["category"].unique().tolist()),
        "intents": sorted(df["intent"].unique().tolist()),
    }
