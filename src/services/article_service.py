from src.database import DatabaseManager
from src.entity.article import Article
from src.repositories.article_repository import ArticleRepository


class ArticleService:
    """Service for managing article operations.

    This service provides high-level operations for articles by delegating
    database operations to the ArticleRepository.
    """

    def __init__(self, db_manager: DatabaseManager):
        self.repository = ArticleRepository(db_manager)

    def save_article(self, article: Article) -> None:
        """Save a single article to the database.

        If the article already exists (based on link), it will be updated.
        Otherwise, a new article will be created.

        Parameters
        ----------
        article : Article
            The article object to save.
        """
        self.repository.save(article)

    def save_articles(self, articles: list[Article]) -> None:
        """Save multiple articles to the database efficiently.

        This method uses bulk operations to save multiple articles at once,
        separating new articles from existing ones for optimal performance.

        Parameters
        ----------
        articles : list[Article]
            List of article objects to save.
        """
        self.repository.save_bulk(articles)

    def get_article_count(self) -> int:
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
