"""
CLI smoke test for the Fashion Finder query pipeline.

Loads a test image, runs the full query pipeline (segmentation → feature
extraction → Pinecone search), and prints a ranked results table.

Usage:
  uv run python scripts/test_query.py --image path/to/shirt.jpg
  uv run python scripts/test_query.py --image shirt.jpg --category "long sleeve top" --season winter --top-k 5
"""
import argparse
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fashion Finder — query pipeline smoke test",
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to a test clothing image (JPEG / PNG)",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Optional category_name filter (e.g. 'long sleeve top')",
    )
    parser.add_argument(
        "--season",
        default=None,
        choices=["spring", "summer", "autumn", "winter"],
        help="Optional season filter",
    )
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Validate image path
    # ------------------------------------------------------------------
    if not args.image.exists():
        print(f"[ERROR] Image not found: {args.image}")
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Load image
    # ------------------------------------------------------------------
    from PIL import Image
    print(f"Loading image: {args.image}")
    pil_image = Image.open(args.image).convert("RGB")
    print(f"  Size: {pil_image.size[0]}×{pil_image.size[1]} px")

    # ------------------------------------------------------------------
    # Build FilterParams
    # ------------------------------------------------------------------
    from src.pipeline.query_pipeline import FilterParams
    filters = FilterParams(
        category_name=args.category,
        season=args.season,
        top_k=args.top_k,
    )
    print(f"\nFilters: category={filters.category_name!r}  "
          f"season={filters.season!r}  top_k={filters.top_k}")

    # ------------------------------------------------------------------
    # Run query
    # ------------------------------------------------------------------
    print("\nRunning query pipeline …")
    t0 = time.perf_counter()

    from src.pipeline.query_pipeline import run_query
    results = run_query(pil_image, filters)

    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Print results table
    # ------------------------------------------------------------------
    col_widths = {
        "rank": 4,
        "score": 7,
        "category": 24,
        "formality": 9,
        "season": 8,
        "image_path": 40,
    }

    header = (
        f"{'Rank':>{col_widths['rank']}} | "
        f"{'Score':>{col_widths['score']}} | "
        f"{'Category':<{col_widths['category']}} | "
        f"{'Formality':>{col_widths['formality']}} | "
        f"{'Season':<{col_widths['season']}} | "
        f"{'Image Path':<{col_widths['image_path']}}"
    )
    separator = "-" * len(header)

    print(f"\n{header}")
    print(separator)

    if not results:
        print("  (no results returned)")
    else:
        for rank, r in enumerate(results, start=1):
            path_display = r.image_path
            if len(path_display) > col_widths["image_path"]:
                path_display = "…" + path_display[-(col_widths["image_path"] - 1):]
            print(
                f"{rank:>{col_widths['rank']}} | "
                f"{r.score:>{col_widths['score']}.4f} | "
                f"{r.category_name:<{col_widths['category']}} | "
                f"{r.formality:>{col_widths['formality']}.2f} | "
                f"{r.season:<{col_widths['season']}} | "
                f"{path_display:<{col_widths['image_path']}}"
            )

    print(separator)
    print(f"\n✓ Done in {elapsed:.2f}s  ({len(results)} results)\n")


if __name__ == "__main__":
    main()
