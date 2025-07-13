import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import spacy
from tqdm import tqdm

from src.const import ROOT
from src.database import DatabaseManager
from src.entity.pos_ngram_similarity import CalculationConfig, PosNgramSimilarityResult
from src.metrics.ngram_cosine import pos_ngram_cosine_similarity
from src.services import ArticleService, PosNgramSimilarityService
from src.visualize_similarity import (
    SimilarityPair,
    analyze_article_similarities,
    create_median_similarity_distribution,
    export_article_similarity_stats_csv,
    visualize_similarities,
)


def convert_to_similarity_pairs(
    similarity_service: PosNgramSimilarityService,
    article_links: list[str],
    model: str,
    ngram_size: int,
    embedding_method: str,
) -> list[SimilarityPair]:
    """Convert similarity data to SimilarityPair objects for visualization using service layer.

    Parameters
    ----------
    similarity_service : PosNgramSimilarityService
        Service instance for accessing similarity data
    article_links : list[str]
        List of article links to consider for visualization
    model : str
        Model used for similarity calculation
    ngram_size : int
        Size of N-grams used
    embedding_method : str
        Method used for embedding

    Returns
    -------
    list[SimilarityPair]
        List of similarity pairs suitable for visualization
    """
    similarity_tuples = similarity_service.get_similarity_pairs_for_articles(
        article_links=article_links, model=model, ngram_size=ngram_size, embedding_method=embedding_method
    )

    similarity_pairs = []
    for article_id_a, article_id_b, similarity_score in similarity_tuples:
        similarity_pairs.append(
            SimilarityPair(
                article_id_a=article_id_a,
                article_id_b=article_id_b,
                ngram_similarity=similarity_score,
            )
        )

    return similarity_pairs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Article similarity analysis script.")
    parser.add_argument(
        "--xml",
        type=Path,
        required=True,
        help="Path to the XML file containing articles.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ja_core_news_sm",
        help="SpaCy model name for POS tagging.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="Size of the n-grams for similarity calculation.",
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="bow",
        choices=["bow", "tfidf"],
        help="Type of embedding to use for n-grams. Currently supports 'bow' (Bag-of-Words) and 'tfidf'.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "output",
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--creator",
        type=str,
        required=True,
        help="Creator name to filter articles by.",
    )
    parser.add_argument(
        "--size",
        type=int,
        help="Size of the maximum number of articles to process. If not specified, all articles will be processed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    xml_file = args.xml
    model_name = args.model
    ngram_size = args.n
    output_dir = args.output_dir
    embedding_type = args.embedding_type
    creator = args.creator
    output_dir.mkdir(parents=True, exist_ok=True)

    db_manager = DatabaseManager()
    article_service = ArticleService(db_manager)
    similarity_service = PosNgramSimilarityService(db_manager)

    # 記事リストを取得
    article_list = article_service.get_articles_by_creator(creator=creator, newest_first=True)
    print(f"Found {len(article_list)} articles.")

    # サイズ制限が指定されている場合は、記事リストを制限
    if args.size is not None:
        article_list = article_list[: args.size]
        print(f"Limited to {len(article_list)} articles based on size parameter.")

    # 本文が100文字未満の記事は除外
    total_articles = len(article_list)
    article_list = [art for art in article_list if len(art.content.markdown) >= 100]
    filtered_articles = len(article_list)
    print(f"Filtered articles (length >= 100): {filtered_articles} articles.")

    # 設定を保存
    config = CalculationConfig(
        creator=creator,
        model=model_name,
        ngram_size=ngram_size,
        embedding_method=embedding_type,
        xml_file=xml_file,
        total_articles_found=total_articles,
        articles_after_filtering=filtered_articles,
    )

    save_dir = output_dir / creator / config.execution_timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = config.save_to_json(save_dir)
    print(f"Configuration saved to: {config_path}")

    article_pairs = list(combinations(article_list, 2))

    # 既存の類似度を確認し、未計算のペアを特定
    pair_links = [(a.link, b.link) for a, b in article_pairs]

    # Use the new missing pairs detection method
    missing_pairs = similarity_service.get_missing_similarities(
        pair_links, config.model, config.ngram_size, config.embedding_method
    )

    # Convert missing pair links back to article objects
    missing_link_set = set(missing_pairs)
    new_pairs_to_calculate = []
    for a, b in article_pairs:
        if (a.link, b.link) in missing_link_set or (b.link, a.link) in missing_link_set:
            new_pairs_to_calculate.append((a, b))

    # モデルの読み込み
    nlp = spacy.load(config.model)

    # 新しい類似度を計算して保存
    newly_calculated_similarities = []
    if new_pairs_to_calculate:
        with tqdm(total=len(new_pairs_to_calculate)) as pb:
            for article_a, article_b in new_pairs_to_calculate:
                similarity = pos_ngram_cosine_similarity(
                    article_a.content.markdown,
                    article_b.content.markdown,
                    n=config.ngram_size,
                    nlp=nlp,
                    embedding_type=config.embedding_method,
                )

                # Create similarity results for both directions
                similarity_result_a_to_b = PosNgramSimilarityResult(
                    other_article_id=article_b.link,
                    model=config.model,
                    ngram_size=config.ngram_size,
                    embedding_method=config.embedding_method,
                    ngram_similarity=similarity,
                )

                similarity_result_b_to_a = PosNgramSimilarityResult(
                    other_article_id=article_a.link,
                    model=config.model,
                    ngram_size=config.ngram_size,
                    embedding_method=config.embedding_method,
                    ngram_similarity=similarity,
                )

                # Save both directions
                similarity_service.save(
                    [(article_a.link, similarity_result_a_to_b), (article_b.link, similarity_result_b_to_a)]
                )
                newly_calculated_similarities.extend([similarity_result_a_to_b, similarity_result_b_to_a])
                # newly_calculated_similarities is now handled above
                pb.update(1)
    else:
        print("No new similarities to calculate.")

    # --- 可視化を実行 ---
    print("Creating visualizations...")

    # 記事リンクを収集（article_listの順序を維持）
    article_links_for_vis = [art.link for art in article_list]

    # 全ての類似度データをSimilarityPair形式に変換
    similarity_pairs = convert_to_similarity_pairs(
        similarity_service=similarity_service,
        article_links=article_links_for_vis,
        model=config.model,
        ngram_size=config.ngram_size,
        embedding_method=config.embedding_method,
    )

    visualize_similarities(
        similarities=similarity_pairs,
        article_links=article_links_for_vis,
        save_dir=save_dir,
    )

    # --- 記事別類似度分析を実行 ---
    print("Creating article-wise similarity analysis...")

    # 記事別の類似度統計を計算
    article_stats = analyze_article_similarities(similarity_pairs, article_links_for_vis, article_list)

    if article_stats:
        # 記事別類似度統計をCSVに出力
        csv_path = save_dir / "article_similarity_statistics.csv"
        export_article_similarity_stats_csv(article_stats, csv_path)
        print(f"Article similarity statistics saved to: {csv_path}")

        # 記事ごとの類似度中央値の分布
        median_dist_path = save_dir / "median_similarity_distribution.png"
        median_dist_title = "Distribution of Article Similarity Medians"

        create_median_similarity_distribution(article_stats, median_dist_path, median_dist_title)
        print(f"Median similarity distribution saved to: {median_dist_path}")

        # 統計サマリーを出力
        print("\nAnalysis Summary:")
        print(f"Total articles analyzed: {len(article_stats)}")
        medians = [stats.median for stats in article_stats]
        print(f"Overall median similarity - Mean: {np.mean(medians):.3f}, Std: {np.std(medians):.3f}")
        print(f"Range of medians: {np.min(medians):.3f} - {np.max(medians):.3f}")
    else:
        print("No similarity data available for analysis.")

    print("All visualizations and analysis completed.")
