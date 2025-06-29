from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm

from src.const import ROOT
from src.metrics.ngram_cosine import pos_ngram_cosine_similarity

from src.database import DatabaseManager
from src.services import ArticleService, PosNgramSimilarityService
from src.entity.pos_ngram_similarity import PosNgramSimilarityResult
import argparse


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

    n_articles = len(article_list)
    heat_map = [[0.0 for _ in range(n_articles)] for _ in range(n_articles)]
    similarity_list = []
    similarity_results = []

    loop_count = int(
        (n_articles * n_articles - n_articles) / 2
    )  # 対角要素を除く組み合わせ数
    pb = tqdm(total=loop_count)
    for i in range(n_articles):
        heat_map[i][i] = 1.0  # 対角要素は自己類似度 1
        for j in range(i + 1, n_articles):
            article_a = article_list[i]
            article_b = article_list[j]
            
            # 結果がDBにすでに存在する場合はスキップ
            existing_similarity = similarity_service.get_similarity(
                article_a.link, article_b.link, model_name, ngram_size, embedding_type
            )
            
            if existing_similarity is not None:
                similarity = existing_similarity
            else:
                similarity = pos_ngram_cosine_similarity(
                    article_a.content.markdown,
                    article_b.content.markdown,
                    n=ngram_size,
                    spacy_model=model_name,
                    embedding_type=embedding_type
                )
                
                # 類似度結果を都度保存（途中で失敗しても成功分は保存したい）
                similarity_result = PosNgramSimilarityResult(
                    article_id_a=article_a.link,
                    article_id_b=article_b.link,
                    model=model_name,
                    ngram_size=ngram_size,
                    embedding_method=embedding_type,
                    ngram_similarity=similarity
                )
                similarity_service.save_similarity(similarity_result)
            
            heat_map[i][j] = similarity
            heat_map[j][i] = similarity
            similarity_list.append(similarity)
            pb.update(1)


    # --- ヒートマップ描画 ---
    # 軸ラベル
    labels = [
        f"{idx + 1}:{Path(art.link).name}" for idx, art in enumerate(article_list)
    ]

    fig, ax = plt.subplots(
        figsize=(max(6, int(n_articles * 0.4)), max(6, int(n_articles * 0.4)))
    )
    im = ax.imshow(heat_map, cmap="hot", interpolation="nearest")

    # 軸に記事ラベルを設定
    ax.set_xticks(range(n_articles))
    ax.set_yticks(range(n_articles))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # カラーバーとタイトルなど
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Article Similarity Heatmap")
    ax.set_xlabel("Articles")
    ax.set_ylabel("Articles")

    fig.tight_layout()
    plt.savefig(str(output_dir / "similarity_heatmap.png"), dpi=300)

    # --- 類似度分布のヒストグラム ---
    plt.figure(figsize=(8, 4))
    plt.hist(similarity_list, bins=n_articles, color="blue", alpha=0.7)
    plt.title("Similarity Distribution")
    plt.xlabel("Similarity")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.savefig(str(output_dir / "similarity_distribution.png"), dpi=300)
