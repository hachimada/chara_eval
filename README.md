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
| post_id     | int  | 記事のID            |
| post_type   | text | 記事のタイプ           |
| status      | text | 記事のステータス         |

- pos_ngram_similarity table

| Column Name      | Type | Description   |
|------------------|------|---------------|
| id               | int  | ID(pkey)      |
| article_id_a     | int  | 記事のID         |
| article_id_b     | int  | 記事のID         |
| model            | text | 使用したモデル名      |
| ngram_size       | int  | ngramのサイズ     |
| embedding_method | text | POSの埋め込み方法    |
| ngram_similarity | text | pos_ngramの類似度 |


### Writing style similarity

To analyze the similarity of writing styles in articles, you can use the `calculate_article_similarities` module.
This module computes the similarity between articles based on their POS n-grams.

The script automatically:

- Skips similarity calculations for article pairs that already exist in the database
- Calculates similarities only for new article combinations
- Creates visualizations (heatmap and distribution plots) after processing

```bash
uv run python -m src.calculate_article_similarities \
--xml datasets/*.xml \
--model ja_core_news_md \
--n 2 \
--embedding_type bow \
--creator <creator_name>
```

Results will be saved in `pos_ngram_similarity` table and visualization files will be generated in the output directory.

#### Outputs

After running the script, the following files will be generated in the `output/{creator}/{timestamp}` directory:

- calculation_config.json - 計算設定
- article_similarity_statistics.csv - 統計CSV
- median_similarity_distribution.png - 分布図

structure of `article_similarity_statistics.csv` is as follows:

| Column Name         | Data Type | Description                                                              |
|---------------------|-----------|--------------------------------------------------------------------------|
| `article_id`        | string    | Article ID (URL)                                                         |
| `article_name`      | string    | Article name (extracted from URL)                                        |
| `pub_date`          | string    | Publication date (ISO format)                                            |
| `character_count`   | int       | Character count of the article                                           |
| `mean_similarity`   | float     | Mean similarity score with other articles (6 decimal places)             |
| `median_similarity` | float     | Median similarity score with other articles (6 decimal places)           |
| `std_similarity`    | float     | Standard deviation of similarity scores (6 decimal places)               |
| `min_similarity`    | float     | Minimum similarity score (6 decimal places)                              |
| `max_similarity`    | float     | Maximum similarity score (6 decimal places)                              |
| `q25_similarity`    | float     | First quartile (25th percentile) of similarity scores (6 decimal places) |
| `q75_similarity`    | float     | Third quartile (75th percentile) of similarity scores (6 decimal places) |
| `num_comparisons`   | int       | Number of articles compared against                                      |

Each row represents statistical information about how similar one article's writing style is compared to all other
articles in the dataset.

## Article Evaluation API

### Overview

The Article Evaluation API provides functionality to evaluate the similarity between a new article and existing articles that have high median similarity scores. This feature is useful for:

- Content quality assessment based on writing style similarity
- Identifying articles with similar writing patterns
- Filtering high-quality reference articles for comparison

### How it works

1. **Filtering**: Load existing article statistics and filter articles by median similarity threshold
2. **Content Loading**: Accept new article content (file path or direct text input)
3. **Similarity Calculation**: Calculate POS n-gram similarity between the new article and filtered existing articles
4. **Statistical Analysis**: Compute statistical measures (mean, median, std, min, max) of similarity scores

### Usage

#### Command Line Interface

**Using direct content:**
```bash
uv run python -m src.eval_article \
--content "新しい記事の内容をここに直接入力" \
--csv output/{creator_name}/{timestamp}/article_similarity_statistics.csv \
--config output/{creator_name}/{timestamp}/calculation_config.json \
--th 0.93
```

**Using file path:**
```bash
uv run python -m src.eval_article \
--file "datasets/new_article.md" \
--csv output/{creator_name}/{timestamp}/article_similarity_statistics.csv \
--config output/{creator_name}/{timestamp}/calculation_config.json \
--th 0.93
```

#### REST API

Start the API server:

```bash
uv run python -m src.api
```

The API will be available at `http://localhost:8000` with the following endpoints:

- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation (Swagger UI)
- `POST /evaluate` - Article evaluation endpoint

#### API Request Examples

**Using direct content:**
```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "ここに新しい記事の内容を直接入力",
    "config_path": "output/creator_name/2024-01-01T12:00:00/calculation_config.json",
    "median_similarity_th": 0.93
  }'
```

**Using file path:**
```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "datasets/new_article.md",
    "config_path": "output/creator_name/2024-01-01T12:00:00/calculation_config.json",
    "median_similarity_th": 0.93
  }'
```

#### Parameters

**API Parameters:**
- `content`: Direct article content text (optional, mutually exclusive with file_path)
- `file_path`: Path to file containing article content (optional, mutually exclusive with content)
- `config_path`: Path to calculation_config.json file (required)
- `median_similarity_th`: Median similarity threshold for filtering existing articles (default: 0.93)

**CLI Parameters:**
- `--content`: Direct article content text (mutually exclusive with --file)
- `--file`: Path to file containing article content (mutually exclusive with --content)
- `--csv`: Path to article_similarity_statistics.csv file (required)
- `--config`: Path to calculation_config.json file (optional)
- `--th`: Median similarity threshold (default: 0.93)

**Important Notes:**
- For API: Either `content` or `file_path` must be provided, but not both
- For CLI: Either `--content` or `--file` must be provided, but not both
- The CSV file (`article_similarity_statistics.csv`) is automatically loaded from the same directory as the config file for API requests
- File paths can be absolute or relative to the current working directory

#### Response

The API returns evaluation results including:

- **Input parameters**: Content length, threshold, file paths, input type (direct_content or file_path)
- **Filtering results**: Total articles, filtered count, filtering ratio
- **Similarity results**: Statistical measures (mean, median, std, min, max, count)
- **Configuration**: Model parameters used for calculation

#### Error Handling

The API now provides clear error messages for:
- Missing or invalid input parameters (both content and file_path provided, or neither provided)
- File not found errors
- Invalid file paths
- Missing configuration or CSV files

