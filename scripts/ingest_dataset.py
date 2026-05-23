"""
CLI entry point for the Fashion Finder dataset ingestion pipeline.

Usage
-----
    uv run python scripts/ingest_dataset.py [--split SPLIT] [--subset N] [--batch-size N]

Examples
--------
    uv run python scripts/ingest_dataset.py --split train --subset 1000
    uv run python scripts/ingest_dataset.py --split validation
"""
import argparse
import logging
import sys
from pathlib import Path

# Make sure the repo root is on sys.path so absolute imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import settings
from src.pipeline.ingest_pipeline import run_ingestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest DeepFashion2 dataset into the Pinecone vector index.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        default="train",
        help="Dataset split to ingest (default: train)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=settings.SUBSET_SIZE,
        help=f"Max annotation files to process (default: {settings.SUBSET_SIZE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.BATCH_SIZE,
        dest="batch_size",
        help=f"Items per Pinecone upsert batch (default: {settings.BATCH_SIZE})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Fashion Finder — Dataset Ingestion")
    logger.info("  split      : %s", args.split)
    logger.info("  subset     : %d", args.subset)
    logger.info("  batch_size : %d", args.batch_size)
    logger.info("  index      : %s", settings.PINECONE_INDEX_NAME)
    logger.info("=" * 60)

    summary = run_ingestion(
        data_split=args.split,
        subset=args.subset,
        batch_size=args.batch_size,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("Ingestion Summary")
    logger.info("  Total items processed : %d", summary["total_processed"])
    logger.info("  Total items upserted  : %d", summary["total_upserted"])
    logger.info("  Total errors          : %d", summary["total_errors"])
    logger.info("  Time elapsed          : %.1f s", summary["elapsed_seconds"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
