"""eval_article.py

Article evaluation functionality for filtering and calculating similarity
with existing articles based on median similarity thresholds.
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import spacy

from src.database import DatabaseManager
from src.entity.article import Article
from src.entity.pos_ngram_similarity import CalculationConfig
from src.metrics.ngram_cosine import pos_ngram_cosine_similarity
from src.services import ArticleService


def load_calculation_config(config_path: Path) -> CalculationConfig:
    """Load calculation configuration from JSON file.

    Parameters
    ----------
    config_path : Path
        Path to the calculation_config.json file

    Returns
    -------
    CalculationConfig
        Loaded configuration object
    """
    return CalculationConfig.from_json(config_path)


def load_article_statistics(csv_path: Path) -> pd.DataFrame:
    """Load article similarity statistics from CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to the article_similarity_statistics.csv file

    Returns
    -------
    pd.DataFrame
        DataFrame containing article statistics
    """
    return pd.read_csv(csv_path)


def filter_articles_by_median_similarity(df: pd.DataFrame, threshold: float = 0.93) -> pd.DataFrame:
    """Filter articles by median similarity threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing article statistics
    threshold : float, optional
        Median similarity threshold, by default 0.93

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only articles above the threshold
    """
    return df[df["median_similarity"] > threshold].copy()


def load_content_from_file_path(file_path: str) -> str:
    """Load content from file path.

    Parameters
    ----------
    file_path : str
        Path to the file containing article content

    Returns
    -------
    str
        Content text

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    """
    content_path = Path(file_path)
    if not content_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not content_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    with open(content_path, "r", encoding="utf-8") as f:
        return f.read()


def get_article_content(content: str | None = None, file_path: str | None = None) -> str:
    """Get article content from either direct content or file path.

    Parameters
    ----------
    content : str, optional
        Direct article content text
    file_path : str, optional
        Path to file containing article content

    Returns
    -------
    str
        Article content

    Raises
    ------
    ValueError
        If neither content nor file_path is provided, or both are provided
    """
    if content is not None and file_path is not None:
        raise ValueError("Provide either content or file_path, not both")
    if content is None and file_path is None:
        raise ValueError("Either content or file_path must be provided")

    if content is not None:
        return content
    else:
        return load_content_from_file_path(file_path)


def calculate_similarity_with_articles(
    new_content: str, articles: list[Article], config: CalculationConfig, nlp: spacy.language.Language
) -> dict[str, Any]:
    """Calculate similarity between new content and articles.

    Parameters
    ----------
    new_content : str
        Content of the new article
    articles : list[Article]
        List of existing articles to compare against
    config : CalculationConfig
        Configuration containing model parameters
    nlp : spacy.language.Language
        Loaded spaCy NLP model for processing

    Returns
    -------
    dict[str, Any]
        Dictionary containing similarity statistics
    """
    similarities = []
    for article in articles:
        # Calculate similarity using the same method as in the original calculation
        similarity = pos_ngram_cosine_similarity(
            new_content,
            article.content.markdown,
            n=config.ngram_size,
            nlp=nlp,
            embedding_type=config.embedding_method,
        )

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


def eval_article_main(
    csv_path: Path,
    config_path: Path,
    median_similarity_th: float = 0.93,
    content: str | None = None,
    file_path: str | None = None,
) -> dict[str, Any]:
    """Evaluate article against filtered existing articles.

    Parameters
    ----------
    csv_path : Path
        Path to article_similarity_statistics.csv
    median_similarity_th : float, optional
        Median similarity threshold, by default 0.93
    config_path : Path, optional
        Path to calculation_config.json, by default None
    content : str, optional
        Direct article content text
    file_path : str, optional
        Path to file containing article content

    Returns
    -------
    dict[str, Any]
        Evaluation results
    """
    # Load new article content
    new_content = get_article_content(content=content, file_path=file_path)

    # Load article statistics
    df = load_article_statistics(csv_path)

    # Filter articles by median similarity threshold
    filtered_articles_df = filter_articles_by_median_similarity(df, median_similarity_th)

    # Get article IDs from filtered articles
    article_ids = filtered_articles_df["article_id"].tolist()

    # Retrieve articles from database
    db_manager = DatabaseManager()
    article_service = ArticleService(db_manager)
    articles = article_service.get_articles_by_ids(article_ids)

    # Load calculation configuration
    config = load_calculation_config(config_path)

    nlp = spacy.load(config.model)

    # Calculate similarities
    similarity_results = calculate_similarity_with_articles(new_content, articles, config, nlp)

    # Prepare results summary
    results = {
        "input_parameters": {
            "content_length": len(new_content),
            "median_similarity_threshold": median_similarity_th,
            "csv_path": str(csv_path),
            "config_path": str(config_path) if config_path else None,
            "input_type": "direct_content" if content is not None else "file_path",
            "file_path": file_path if file_path is not None else None,
        },
        "filtering_results": {
            "total_articles": len(df),
            "filtered_articles": len(filtered_articles_df),
            "filtering_ratio": len(filtered_articles_df) / len(df) if len(df) > 0 else 0,
        },
        "similarity_results": similarity_results,
        "configuration": config.to_dict() if config else None,
    }

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate new article similarity against filtered existing articles.")

    # Content input options (mutually exclusive)
    content_group = parser.add_mutually_exclusive_group(required=True)
    content_group.add_argument("--content", type=str, help="Direct article content text")
    content_group.add_argument("--file", type=str, help="Path to file containing article content")

    parser.add_argument("--csv", type=Path, required=True, help="Path to article_similarity_statistics.csv file")
    parser.add_argument("--th", type=float, default=0.93, help="Median similarity threshold (default: 0.93)")
    parser.add_argument("--config", type=Path, help="Path to calculation_config.json file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = eval_article_main(
        csv_path=args.csv,
        median_similarity_th=args.th,
        config_path=args.config,
        content=args.content,
        file_path=args.file,
    )

    # Print results summary
    print("Evaluation completed:")
    print(f"- Content length: {results['input_parameters']['content_length']} characters")
    print(f"- Threshold: {results['input_parameters']['median_similarity_threshold']}")
    print(f"- Total articles: {results['filtering_results']['total_articles']}")
    print(f"- Filtered articles: {results['filtering_results']['filtered_articles']}")
    print(f"- Filtering ratio: {results['filtering_results']['filtering_ratio']:.3f}")

    # Print similarity statistics
    stats = results["similarity_results"]["statistics"]
    if stats["count"] > 0:
        print("\nSimilarity Statistics:")
        print(f"- Count: {stats['count']}")
        print(f"- Mean: {stats['mean']:.3f}")
        print(f"- Median: {stats['median']:.3f}")
        print(f"- Std: {stats['std']:.3f}")
        print(f"- Min: {stats['min']:.3f}")
        print(f"- Max: {stats['max']:.3f}")
    else:
        print("\nNo articles passed the filtering threshold or similarity calculation failed.")
