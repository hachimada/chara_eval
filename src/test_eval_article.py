"""test_eval_article.py

Test evaluation functionality for all existing articles by treating each as a new article
and calculating similarity statistics against filtered article groups.
"""

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.database import DatabaseManager
from src.entity.article import Article
from src.entity.pos_ngram_similarity import CalculationConfig
from src.eval_article import (
    filter_articles_by_median_similarity,
    load_article_statistics,
    load_calculation_config,
)
from src.services import ArticleService, PosNgramSimilarityService


def get_filtered_articles_excluding_test_article(
    csv_path: Path,
    median_similarity_th: float,
    test_article_id: str,
    article_service: ArticleService,
) -> list[Article]:
    """Get filtered articles excluding the test article.

    Parameters
    ----------
    csv_path : Path
        Path to article_similarity_statistics.csv
    median_similarity_th : float
        Median similarity threshold for filtering
    test_article_id : str
        ID of the test article to exclude
    article_service : ArticleService
        Service for retrieving articles

    Returns
    -------
    list[Article]
        List of filtered articles excluding the test article
    """
    # Load article statistics
    df = load_article_statistics(csv_path)

    # Filter articles by median similarity threshold
    filtered_df = filter_articles_by_median_similarity(df, median_similarity_th)

    # Exclude the test article from filtered results
    filtered_df = filtered_df[filtered_df["article_id"] != test_article_id]

    # Get article IDs from filtered articles
    article_ids = filtered_df["article_id"].tolist()

    # Retrieve articles from database
    articles = article_service.get_articles_by_ids(article_ids)

    return articles


def get_similarity_with_articles(
    new_article_id: str,
    articles: list[Article],
    config: CalculationConfig,
    similarity_service: PosNgramSimilarityService,
) -> dict[str, Any]:
    """Get similarity statistics from pos_ngram_similarity table.

    Parameters
    ----------
    new_article_id : str
        ID of the new article to compare
    articles : list[Article]
        List of articles to compare against
    config : CalculationConfig
        Configuration containing model parameters
    similarity_service : PosNgramSimilarityService
        Service for accessing similarity data

    Returns
    -------
    dict[str, Any]
        Dictionary containing similarity statistics
    """
    similarities = []
    target_article_ids = [article.link for article in articles]

    for target_article_id in target_article_ids:
        # Get similarity from pos_ngram_similarity table
        similarity = similarity_service.get_similarity_between_articles(
            article_id_a=new_article_id,
            article_id_b=target_article_id,
            model=config.model,
            ngram_size=config.ngram_size,
            embedding_method=config.embedding_method,
        )

        if similarity is not None:
            similarities.append(similarity)

    # Calculate statistics
    if similarities:
        statistics = {
            "count": len(similarities),
            "mean": float(np.mean(similarities)),
            "median": float(np.median(similarities)),
            "std": float(np.std(similarities)),
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
        }
    else:
        statistics = {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}

    return {"statistics": statistics}


def test_all_articles_evaluation(
    csv_path: Path,
    config_path: Path,
    median_similarity_th: float = 0.93,
    output_path: Path | None = None,
) -> None:
    """Test evaluation for all existing articles.

    Parameters
    ----------
    csv_path : Path
        Path to article_similarity_statistics.csv
    config_path : Path
        Path to calculation_config.json
    median_similarity_th : float, optional
        Median similarity threshold, by default 0.93
    output_path : Path, optional
        Path to output CSV file, by default None
    """
    # Load calculation configuration
    config = load_calculation_config(config_path)

    # Initialize database and services
    db_manager = DatabaseManager()
    article_service = ArticleService(db_manager)
    similarity_service = PosNgramSimilarityService(db_manager)

    # Get all articles
    all_articles = article_service.get_all_articles()

    # Prepare results list
    results = []

    print(f"Testing {len(all_articles)} articles...")

    for test_article in tqdm(all_articles, desc="Processing articles"):
        # Get filtered articles excluding the test article
        filtered_articles = get_filtered_articles_excluding_test_article(
            csv_path=csv_path,
            median_similarity_th=median_similarity_th,
            test_article_id=test_article.link,
            article_service=article_service,
        )

        if not filtered_articles:
            tqdm.write(f"  No articles passed filtering for test article {test_article.link}")
            # Record with null statistics
            results.append(
                {
                    "article_id": test_article.link,
                    "count": 0,
                    "mean": None,
                    "median": None,
                    "std": None,
                    "min": None,
                    "max": None,
                }
            )
            continue

        # Get similarity statistics from pos_ngram_similarity table
        similarity_results = get_similarity_with_articles(
            new_article_id=test_article.link,
            articles=filtered_articles,
            config=config,
            similarity_service=similarity_service,
        )

        # Extract statistics
        stats = similarity_results["statistics"]

        # Create result record
        result_record = {
            "article_id": test_article.link,
            "count": stats["count"],
            "mean": stats["mean"],
            "median": stats["median"],
            "std": stats["std"],
            "min": stats["min"],
            "max": stats["max"],
        }

        results.append(result_record)

        if stats["median"] is not None:
            tqdm.write(
                f"  Completed {test_article.link}: {stats['count']} comparisons, median similarity: {stats['median']:.3f}"
            )
        else:
            tqdm.write(f"  Completed {test_article.link}: {stats['count']} comparisons, no valid similarities found")

    # Output results to CSV
    if output_path is None:
        output_path = Path("article_evaluation_results.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["article_id", "article_link", "count", "mean", "median", "std", "min", "max"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(f"\nResults saved to: {output_path}")
    print(f"Total articles processed: {len(results)}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test evaluation for all existing articles by treating each as new article."
    )

    parser.add_argument("--csv", type=Path, required=True, help="Path to article_similarity_statistics.csv file")
    parser.add_argument("--config", type=Path, required=True, help="Path to calculation_config.json file")
    parser.add_argument("--th", type=float, default=0.93, help="Median similarity threshold (default: 0.93)")
    parser.add_argument("--output", type=Path, help="Path to output CSV file (default: article_evaluation_results.csv)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    test_all_articles_evaluation(
        csv_path=args.csv,
        config_path=args.config,
        median_similarity_th=args.th,
        output_path=args.output,
    )
