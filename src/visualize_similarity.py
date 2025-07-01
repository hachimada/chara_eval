from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.const import ROOT


@dataclass
class SimilarityPair:
    """Represents a similarity relationship between two articles."""

    article_id_a: str
    article_id_b: str
    ngram_similarity: float


def create_similarity_matrix(
    similarities: List[SimilarityPair], article_links: List[str]
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Create a similarity matrix from a list of similarity records.

    Parameters
    ----------
    similarities : List[SimilarityPair]
        A list of similarity records. The order of this list does not matter.
    article_links : List[str]
        A list of unique article links. The order of this list defines the
        order of rows and columns in the matrix.

    Returns
    -------
    np.ndarray
        An n x n NumPy array representing the similarity matrix.
    Dict[str, int]
        A dictionary mapping each article link to its corresponding index,
        preserving the order.

    Notes
    -----
    The order of the matrix's rows and columns is explicitly determined by the
    order of the `article_links` list. This ensures that the resulting
    heatmap labels correspond to the intended article sequence.
    """
    n_articles = len(article_links)
    link_to_idx = {link: idx for idx, link in enumerate(article_links)}
    matrix = np.eye(n_articles)

    for sim in similarities:
        if sim.article_id_a in link_to_idx and sim.article_id_b in link_to_idx:
            idx_a = link_to_idx[sim.article_id_a]
            idx_b = link_to_idx[sim.article_id_b]
            matrix[idx_a, idx_b] = sim.ngram_similarity
            matrix[idx_b, idx_a] = sim.ngram_similarity

    return matrix, link_to_idx


def create_heatmap(
    similarity_matrix: np.ndarray,
    article_labels: List[str],
    output_path: Path,
    title: str = "Article Similarity Heatmap",
):
    """ヒートマップを作成して保存"""
    n_articles = len(article_labels)
    fig_size = max(8, int(n_articles * 0.4))

    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        similarity_matrix,
        xticklabels=article_labels,
        yticklabels=article_labels,
        annot=False,
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
    )

    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("Articles", fontsize=12)
    plt.ylabel("Articles", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_similarity_distribution(
    similarities: List[SimilarityPair], output_path: Path, title: str = "Similarity Distribution"
):
    """類似度分布のヒストグラムを作成"""
    similarity_values = [sim.ngram_similarity for sim in similarities]

    plt.figure(figsize=(10, 6))
    plt.hist(similarity_values, bins=int(len(similarities) * 0.5), color="lightblue", alpha=0.7, edgecolor="black")
    plt.title(title, fontsize=14)
    plt.xlabel("Similarity Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis="y", alpha=0.3)

    # 統計情報を追加
    mean_sim = np.mean(similarity_values)
    median_sim = np.median(similarity_values)
    std_sim = np.std(similarity_values)

    plt.axvline(mean_sim, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_sim:.3f}")
    plt.axvline(median_sim, color="green", linestyle="--", linewidth=2, label=f"Median: {median_sim:.3f}")
    plt.legend()

    # テキストボックスで統計情報を表示
    stats_text = f"Mean: {mean_sim:.3f}\nMedian: {median_sim:.3f}\nStd: {std_sim:.3f}\nCount: {len(similarity_values)}"
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def visualize_similarities(
    similarities: List[SimilarityPair],
    article_links: List[str],
    creator: str,
    model: str,
    ngram_size: int,
    embedding_method: str,
    output_dir: Path = ROOT / "output",
):
    """類似度データを可視化"""
    save_dir = output_dir / creator
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Visualizing {len(similarities)} similarity records for {len(article_links)} articles.")

    # 記事ラベルを作成（記事IDから簡潔な名前を生成）
    article_labels = []
    for i, link in enumerate(article_links):
        article_name = Path(link).name if Path(link).name else f"Article_{i + 1}"
        article_labels.append(f"{i + 1}:{article_name}")

    # 類似度マトリックスを作成
    similarity_matrix, link_to_idx = create_similarity_matrix(similarities, article_links)

    # ファイル名に使用するサフィックス
    suffix_parts = [f"model_{model}", f"n_{ngram_size}", f"emb_{embedding_method}"]
    suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

    # ヒートマップを作成
    heatmap_path = save_dir / f"similarity_heatmap{suffix}.png"
    title = "Article Similarity Heatmap"
    if suffix_parts:
        title += f" ({', '.join(suffix_parts)})"

    create_heatmap(similarity_matrix, article_labels, heatmap_path, title)
    print(f"Heatmap saved to: {heatmap_path}")

    # 類似度分布を作成
    distribution_path = save_dir / f"similarity_distribution{suffix}.png"
    dist_title = "Similarity Distribution"
    if suffix_parts:
        dist_title += f" ({', '.join(suffix_parts)})"

    create_similarity_distribution(similarities, distribution_path, dist_title)
    print(f"Distribution plot saved to: {distribution_path}")


@dataclass
class ArticleSimilarityStats:
    """Statistics for similarity distribution of a single article."""

    article_id: str
    similarities: list[float]
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    q25: float
    q75: float
    character_count: int


def analyze_article_similarities(
    similarities: List[SimilarityPair], article_links: List[str], articles: List = None
) -> List[ArticleSimilarityStats]:
    """Analyze similarity distributions for each article.

    Parameters
    ----------
    similarities : List[SimilarityPair]
        List of similarity pairs
    article_links : List[str]
        List of article links to analyze
    articles : List, optional
        List of article objects with content information

    Returns
    -------
    List[ArticleSimilarityStats]
        List of similarity statistics for each article
    """
    article_stats = []

    # Create mapping from article_id to character count
    char_count_map = {}
    if articles:
        for article in articles:
            char_count_map[article.link] = len(article.content.markdown)

    for article_id in article_links:
        # Get similarities for this article (both as source and target)
        article_similarities = []

        for sim in similarities:
            if sim.article_id_a == article_id:
                article_similarities.append(sim.ngram_similarity)
            elif sim.article_id_b == article_id:
                article_similarities.append(sim.ngram_similarity)

        if article_similarities:
            similarities_array = np.array(article_similarities)

            character_count = char_count_map.get(article_id, 0)

            stats = ArticleSimilarityStats(
                article_id=article_id,
                similarities=article_similarities,
                mean=float(np.mean(similarities_array)),
                median=float(np.median(similarities_array)),
                std=float(np.std(similarities_array)),
                min_val=float(np.min(similarities_array)),
                max_val=float(np.max(similarities_array)),
                q25=float(np.percentile(similarities_array, 25)),
                q75=float(np.percentile(similarities_array, 75)),
                character_count=character_count,
            )
            article_stats.append(stats)

    return article_stats


def export_article_similarity_stats_csv(
    article_stats: List[ArticleSimilarityStats],
    output_path: Path,
) -> None:
    """Export article similarity statistics to CSV file.

    Parameters
    ----------
    article_stats : List[ArticleSimilarityStats]
        List of article similarity statistics
    output_path : Path
        Path to save the CSV file
    """
    if not article_stats:
        return

    import csv

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "article_id",
            "article_name",
            "character_count",
            "mean_similarity",
            "median_similarity",
            "std_similarity",
            "min_similarity",
            "max_similarity",
            "q25_similarity",
            "q75_similarity",
            "num_comparisons",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for stats in article_stats:
            # Extract article name from path
            article_name = Path(stats.article_id).name if Path(stats.article_id).name else "Unknown"

            writer.writerow(
                {
                    "article_id": stats.article_id,
                    "article_name": article_name,
                    "character_count": stats.character_count,
                    "mean_similarity": round(stats.mean, 6),
                    "median_similarity": round(stats.median, 6),
                    "std_similarity": round(stats.std, 6),
                    "min_similarity": round(stats.min_val, 6),
                    "max_similarity": round(stats.max_val, 6),
                    "q25_similarity": round(stats.q25, 6),
                    "q75_similarity": round(stats.q75, 6),
                    "num_comparisons": len(stats.similarities),
                }
            )


def create_median_similarity_distribution(
    article_stats: List[ArticleSimilarityStats],
    output_path: Path,
    title: str = "Distribution of Article Similarity Medians",
) -> None:
    """Create distribution of similarity medians across all articles.

    Parameters
    ----------
    article_stats : List[ArticleSimilarityStats]
        List of article similarity statistics
    output_path : Path
        Path to save the visualization
    title : str, default "Distribution of Article Similarity Medians"
        Title for the visualization
    """
    if not article_stats:
        return

    medians = [stats.median for stats in article_stats]

    plt.figure(figsize=(10, 6))

    # Create histogram
    plt.hist(medians, bins=max(5, len(medians) // 3), color="lightcoral", alpha=0.7, edgecolor="black")

    # Add statistics
    overall_mean = np.mean(medians)
    overall_median = np.median(medians)
    overall_std = np.std(medians)

    plt.axvline(overall_mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {overall_mean:.3f}")
    plt.axvline(overall_median, color="green", linestyle="--", linewidth=2, label=f"Median: {overall_median:.3f}")

    plt.title(title, fontsize=14)
    plt.xlabel("Median Similarity Score", fontsize=12)
    plt.ylabel("Number of Articles", fontsize=12)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    # Add statistics text box
    stats_text = (
        f"Mean: {overall_mean:.3f}\nMedian: {overall_median:.3f}\nStd: {overall_std:.3f}\nArticles: {len(medians)}"
    )
    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
