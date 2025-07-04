# note記事分析

[note株式会社](https://note.jp/)が提供するサービス[note](https://note.com/)記事を分析するためのリポジトリです。

## overview

This repository analyzes xml files that contain articles exported from the note service.

## Setup environment

- setup virtual environment and install dependencies

   ```bash
   uv sync
   uv run python -m ensurepip --upgrade
   uv run python -m spacy download ja_core_news_sm
   uv run python -m spacy download ja_core_news_md
   ```
- migrate database

   ```bash
   # migration
   uv run alembic upgrade head
   ```

## How to Use this Repository

### Get xml files and save to db

1. export articles from note service as xml files
2. put xml files in `datasets` directory
3. save articles to database running the following command

    ```bash
    uv run python -m src.load_xml --xml datasets/*.xml
    ```

### Database structure

- article table

| Column Name | Type | Description      |
|-------------|------|------------------|
| id          | text | 記事のURL(pkey)     |
| creator     | text | 作成者のユーザ名         |
| pub_date    | text | 記事の公開日時          |
| post_date   | text | 記事の投稿日時          |
| title       | text | 記事のタイトル          |
| body_md     | text | markdown形式の記事の本文 |
| body_html   | text | HTML形式の記事の本文     |
| post_id     | int  | 記事のID             |
| post_type   | text | 記事のタイプ           |
| status      | text | 記事のステータス         |

- pos_ngram_similarity table

| Column Name      | Type | Description   |
|------------------|------|---------------|
| id               | int  | ID(pkey)      |
| article_id_a     | int  | 記事のID         |
| article_id_b     | int  | 記事のID         |
| model           | text | 使用したモデル名      |
|ngram_size      | int  | ngramのサイズ        |
|embedding_method | text | POSの埋め込み方法      |
| ngram_similarity | text | pos_ngramの類似度 |

- 

### Writing style similarity

To analyze the similarity of writing styles in articles, you can use the `pos_ngram_similarity` module. 
This module computes the similarity between articles based on their POS n-grams.

The script automatically:
- Skips similarity calculations for article pairs that already exist in the database
- Calculates similarities only for new article combinations
- Creates visualizations (heatmap and distribution plots) after processing

```bash
uv run python -m src.pos_ngram_similarity \
--xml datasets/*.xml \
--model ja_core_news_md \
--n 2 \
--embedding_type bow \
--creator <creator_name>
```

Results will be saved in `pos_ngram_similarity` table and visualization files will be generated in the output directory.

