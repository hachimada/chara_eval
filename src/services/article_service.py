from src.database import DatabaseManager
from src.entity.article import Article
from src.repositories.article_repository import ArticleRepository


class ArticleService:
    """Service for managing high-level article operations.

    This service provides business logic and high-level operations for articles,
    abstracting complex operations and ensuring consistent bulk processing.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize service with database manager.

        Parameters
        ----------
        db_manager : DatabaseManager
            Database manager instance for repository initialization.
        """
        self.repository = ArticleRepository(db_manager)

    def save(self, articles: Article | list[Article]) -> None:
        """Save articles to the database using bulk operations.

        This method always uses bulk operations for optimal performance,
        even when saving a single article. It handles both new articles
        and updates to existing articles automatically.

        Parameters
        ----------
        articles : Article | list[Article]
            Single article or list of articles to save.
        """
        article_list = [articles] if isinstance(articles, Article) else articles
        if article_list:
            self.repository.save_bulk(article_list)

    def get_total_count(self) -> int:
        """Get the total number of articles in the database.

        Returns
        -------
        int
            Total count of articles in the database.
        """
        return self.repository.count()

    def get_articles_by_creator(self, creator: str, newest_first: bool = True) -> list[Article]:
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
        return self.repository.find_by_creator(creator, newest_first)

    def get_articles_by_ids(self, article_ids: list[str]) -> list[Article]:
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
        return self.repository.find_by_ids(article_ids)
