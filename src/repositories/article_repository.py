"""Repository for article database operations."""

from datetime import datetime

from src.database import DatabaseManager
from src.entity.article import Article, Content
from src.models import ArticleModel
from src.repositories.base_repository import BaseRepository


class ArticleRepository(BaseRepository[Article]):
    """Repository for managing article database operations.

    This repository handles all database operations related to articles,
    including saving, retrieving, and counting articles.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize repository with database manager.

        Parameters
        ----------
        db_manager : DatabaseManager
            Database manager instance for database operations.
        """
        super().__init__(db_manager)

    def save_bulk(self, articles: list[Article]) -> None:
        """Save multiple articles to the database efficiently.

        This method uses bulk operations to save multiple articles at once,
        separating new articles from existing ones for optimal performance.

        Parameters
        ----------
        articles : list[Article]
            List of article objects to save.
        """
        with self.db_manager.get_session() as session:
            # 既存の記事IDを取得
            existing_ids = set(session.query(ArticleModel.id).all())
            existing_ids = {article_id[0] for article_id in existing_ids}

            # 新規記事と更新記事を分別
            new_articles = []
            update_articles = []

            for article in articles:
                article_data = {
                    "id": article.link,
                    "creator": article.creator,
                    "pub_date": article.pub_date.isoformat() if article.pub_date else "",
                    "post_date": article.post_date.isoformat() if article.post_date else "",
                    "title": article.title,
                    "body_md": article.content.markdown,
                    "body_html": article.content.html,
                    "post_id": article.post_id,
                    "post_type": article.post_type,
                    "status": article.status,
                }

                if article.link in existing_ids:
                    update_articles.append(article_data)
                else:
                    new_articles.append(article_data)

            # 新規記事を一括挿入
            if new_articles:
                session.bulk_insert_mappings(ArticleModel, new_articles)

            # 既存記事を一括更新
            if update_articles:
                session.bulk_update_mappings(ArticleModel, update_articles)

    def count(self) -> int:
        """Get the total number of articles in the database.

        Returns
        -------
        int
            Total count of articles in the database.
        """
        with self.db_manager.get_session() as session:
            return session.query(ArticleModel).count()

    def find_by_creator(self, creator: str, newest_first: bool = True) -> list[Article]:
        """Get all articles by a specific creator.

        Parameters
        ----------
        creator : str
            Name of the creator/author.
        newest_first : bool, optional
            Whether to sort articles by publication date in descending order.
            Default is True.

        Returns
        -------
        list[Article]
            List of articles by the specified creator.
        """
        with self.db_manager.get_session() as session:
            query = session.query(ArticleModel).filter(ArticleModel.creator == creator)
            if newest_first:
                query = query.order_by(ArticleModel.pub_date.desc())
            else:
                query = query.order_by(ArticleModel.pub_date.asc())
            article_models = query.all()

            articles = []
            for model in article_models:
                content = Content(html=model.body_html)
                article = Article(
                    title=model.title,
                    link=model.id,
                    creator=model.creator,
                    content=content,
                    post_id=model.post_id,
                    pub_date=datetime.fromisoformat(model.pub_date) if model.pub_date else datetime.now(),
                    post_date=datetime.fromisoformat(model.post_date) if model.post_date else datetime.now(),
                    post_type=model.post_type,
                    status=model.status,
                )
                articles.append(article)
            return articles

    def find_by_ids(self, article_ids: list[str]) -> list[Article]:
        """Get multiple articles by their IDs.

        Parameters
        ----------
        article_ids : list[str]
            List of article IDs to retrieve.

        Returns
        -------
        list[Article]
            List of found articles. Articles not found are omitted from the result.
        """
        with self.db_manager.get_session() as session:
            models = session.query(ArticleModel).filter(ArticleModel.id.in_(article_ids)).all()

            articles = []
            for model in models:
                content = Content(html=model.body_html)
                article = Article(
                    title=model.title,
                    link=model.id,
                    creator=model.creator,
                    content=content,
                    post_id=model.post_id,
                    pub_date=datetime.fromisoformat(model.pub_date) if model.pub_date else datetime.now(),
                    post_date=datetime.fromisoformat(model.post_date) if model.post_date else datetime.now(),
                    post_type=model.post_type,
                    status=model.status,
                )
                articles.append(article)
            return articles
