from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.const import ROOT


def create_similarity_matrix(similarities: List, article_links: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Create a similarity matrix from a list of similarity records.

    Parameters
    ----------
    similarities : List
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


def create_similarity_distribution(similarities: List, output_path: Path, title: str = "Similarity Distribution"):
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
    similarities: List,
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
