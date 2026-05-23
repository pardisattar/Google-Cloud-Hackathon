"""
Smoke test — verify Pinecone index after ingestion.

Usage
-----
    uv run python scripts/verify_index.py

Prints
------
  • Index stats (vector count, dimension)
  • Top-5 results for a random 583-d query vector
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config.settings import settings
from src.vector_store.pinecone_client import _get_client, _get_index, init_index


def main() -> None:
    print("\n" + "=" * 60)
    print("Fashion Finder — Index Verification")
    print("=" * 60)

    # Ensure index exists
    init_index()

    pc = _get_client()
    index = _get_index()

    # --- Index stats ---
    stats = index.describe_index_stats()
    print(f"\nIndex : {settings.PINECONE_INDEX_NAME}")
    print(f"  Total vector count : {stats.total_vector_count:,}")
    print(f"  Dimension          : {stats.dimension}")

    if stats.total_vector_count == 0:
        print("\n⚠  Index is empty — run ingestion first.")
        return

    # --- Random query ---
    dim = 583
    rng = np.random.default_rng(seed=42)
    query_vec = rng.standard_normal(dim).astype(np.float32)
    query_vec /= np.linalg.norm(query_vec) + 1e-8

    print(f"\nQuerying with a random {dim}-d unit vector (seed=42) …")
    response = index.query(
        vector=query_vec.tolist(),
        top_k=5,
        include_metadata=True,
    )

    print("\nTop-5 results:")
    print(f"  {'Rank':<6} {'Score':>8}  {'ID':<30}  Category")
    print("  " + "-" * 70)
    for rank, match in enumerate(response.matches, start=1):
        cat = (match.metadata or {}).get("category_name", "N/A")
        print(f"  {rank:<6} {match.score:>8.4f}  {match.id:<30}  {cat}")

    print("\n✅  Index verification complete.\n")


if __name__ == "__main__":
    main()
