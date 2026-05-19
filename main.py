"""Entry point for the customer-service data analyst agent.

Phase 1 today: prints a summary of the loaded Bitext dataset.
Will be replaced in Phase 5 with the interactive CLI (``--session`` REPL).
"""

import logging

from dotenv import load_dotenv

from agent.data import summarize


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    info = summarize()
    categories: list[str] = info["categories"]  # type: ignore[assignment]
    intents: list[str] = info["intents"]  # type: ignore[assignment]

    print(f"Rows:       {info['row_count']:,}")
    print(f"Categories ({len(categories)}): {', '.join(categories)}")
    print(f"Intents    ({len(intents)}):")
    for intent in intents:
        print(f"  - {intent}")


if __name__ == "__main__":
    main()
