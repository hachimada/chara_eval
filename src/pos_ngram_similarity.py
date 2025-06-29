import argparse
from itertools import combinations
from pathlib import Path

from tqdm import tqdm

from src.const import ROOT
from src.database import DatabaseManager
from src.entity.pos_ngram_similarity import PosNgramSimilarityResult
from src.metrics.ngram_cosine import pos_ngram_cosine_similarity
from src.services import ArticleService, PosNgramSimilarityService
from src.visualize_similarity import visualize_similarities


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

    # 本文が100文字未満の記事は除外
    article_list = [art for art in article_list if len(art.content.markdown) >= 100]
    print(f"Filtered articles (length >= 100): {len(article_list)} articles.")

    article_pairs = list(combinations(article_list, 2))

    # 既存の類似度を一括取得

    pair_links = [(a.link, b.link) for a, b in article_pairs]
    existing_similarities = similarity_service.get_similarities_by_pairs(
        pair_links, model_name, ngram_size, embedding_type
    )

    existing_pairs = set()
    for sim in existing_similarities:
        existing_pairs.add(tuple(sorted((sim.article_id_a, sim.article_id_b))))

    # 未計算のペアを特定
    new_pairs_to_calculate = []
    for article_a, article_b in article_pairs:
        pair_tuple = tuple(sorted((article_a.link, article_b.link)))
        if pair_tuple not in existing_pairs:
            new_pairs_to_calculate.append((article_a, article_b))

    # 新しい類似度を計算して保存
    newly_calculated_similarities = []
    if new_pairs_to_calculate:
        with tqdm(total=len(new_pairs_to_calculate)) as pb:
            for article_a, article_b in new_pairs_to_calculate:
                similarity = pos_ngram_cosine_similarity(
                    article_a.content.markdown,
                    article_b.content.markdown,
                    n=ngram_size,
                    spacy_model=model_name,
                    embedding_type=embedding_type,
                )

                similarity_result = PosNgramSimilarityResult(
                    article_id_a=article_a.link,
                    article_id_b=article_b.link,
                    model=model_name,
                    ngram_size=ngram_size,
                    embedding_method=embedding_type,
                    ngram_similarity=similarity,
                )

                similarity_service.save_similarity(similarity_result)
                newly_calculated_similarities.append(similarity_result)
                pb.update(1)
    else:
        print("No new similarities to calculate.")

    # --- 可視化を実行 ---
    print("Creating visualizations...")

    # 全ての類似度データを結合
    all_similarities = existing_similarities + newly_calculated_similarities

    # 記事リンクを収集（article_listの順序を維持）
    article_links_for_vis = [art.link for art in article_list]

    visualize_similarities(
        similarities=all_similarities,
        article_links=article_links_for_vis,
        model=model_name,
        ngram_size=ngram_size,
        embedding_method=embedding_type,
        creator=creator,
        output_dir=output_dir,
    )
    print("Visualizations completed.")
