"""Entry point for the customer-service data analyst agent.

Parses ``--session`` and drops into the interactive CLI (see agent/cli.py).
"""

import argparse
import logging
import uuid

from dotenv import load_dotenv

from agent.cli import run


def main() -> None:
    load_dotenv()
    # Quiet the noisy HF / httpx INFO logs; the CLI prints its own trace.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Customer Service Data Analyst Agent")
    parser.add_argument(
        "--session",
        default=None,
        help="Session id. Reuse the same id to continue a conversation. "
        "Defaults to a random id.",
    )
    args = parser.parse_args()

    session_id = args.session or f"session-{uuid.uuid4().hex[:8]}"
    run(session_id)


if __name__ == "__main__":
    main()
