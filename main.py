import json
import logging

from datasets import Dataset, load_dataset
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


def load_bitext_dataset() -> Dataset:
    dataset = load_dataset(DATASET_ID, split="train")
    logger.debug("Loaded %s with %d rows", DATASET_ID, len(dataset))
    logger.debug("Sample row:\n%s", json.dumps(dataset[0], indent=2, ensure_ascii=False))
    logger.debug("Sample row:\n%s", json.dumps(dataset[1], indent=2, ensure_ascii=False))
    logger.debug("Sample row:\n%s", json.dumps(dataset[2], indent=2, ensure_ascii=False))
    return dataset


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.DEBUG)
    load_bitext_dataset()


if __name__ == "__main__":
    main()
