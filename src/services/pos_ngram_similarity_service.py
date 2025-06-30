from typing import Optional

from src.database import DatabaseManager
from src.entity.pos_ngram_similarity import PosNgramSimilarityResult
from src.repositories.pos_ngram_similarity_repository import PosNgramSimilarityRepository


class PosNgramSimilarityService:
    """Service for managing high-level POS N-gram similarity operations.

    This service provides business logic and high-level operations for POS N-gram
    similarity analysis, abstracting complex operations and ensuring consistent
    bulk processing for optimal performance.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize service with database manager.

        Parameters
        ----------
        db_manager : DatabaseManager
            Database manager instance for repository initialization.
        """
        self.repository = PosNgramSimilarityRepository(db_manager)

    def save(
        self, article_similarities: tuple[str, PosNgramSimilarityResult] | list[tuple[str, PosNgramSimilarityResult]]
    ) -> None:
        """Save similarity results to the database using bulk operations.

        This method always uses bulk operations for optimal performance,
        even when saving a single result. It handles deduplication and
        validation automatically.

        Parameters
        ----------
        article_similarities : tuple[str, PosNgramSimilarityResult] | list[tuple[str, PosNgramSimilarityResult]]
            Single tuple or list of tuples containing (article_id, similarity_result).
        """
        similarity_list = [article_similarities] if isinstance(article_similarities, tuple) else article_similarities
        if similarity_list:
            self.repository.save_bulk(similarity_list)

    def get_similarities_with_filters(
        self, model: Optional[str] = None, ngram_size: Optional[int] = None, embedding_method: Optional[str] = None
    ) -> list[PosNgramSimilarityResult]:
        """Get similarity results with optional filtering criteria.

        This method provides a high-level interface for retrieving similarity
        results with flexible filtering options.

        Parameters
        ----------
        model : Optional[str], default None
            Filter by model if provided
        ngram_size : Optional[int], default None
            Filter by N-gram size if provided
        embedding_method : Optional[str], default None
            Filter by embedding method if provided

        Returns
        -------
        list[PosNgramSimilarityResult]
            List of similarity results matching the filters
        """
        return self.repository.find_all(model, ngram_size, embedding_method)

    def get_missing_similarities(
        self, pairs: list[tuple[str, str]], model: str, ngram_size: int, embedding_method: str
    ) -> list[tuple[str, str]]:
        """Identify article pairs that don't have similarity calculations yet.

        This method provides business logic to determine which article pairs
        need similarity calculation, useful for incremental processing.

        Parameters
        ----------
        pairs : list[tuple[str, str]]
            List of article ID pairs to check
        model : str
            Model used for similarity calculation
        ngram_size : int
            Size of N-grams used
        embedding_method : str
            Method used for embedding

        Returns
        -------
        list[tuple[str, str]]
            List of pairs that don't have similarity calculations
        """
        missing_pairs = []
        for pair in pairs:
            if not self.repository.exists_for_article(pair[0], pair[1], model, ngram_size, embedding_method):
                missing_pairs.append(pair)
        return missing_pairs

    def get_similarity_pairs_for_articles(
        self,
        article_links: list[str],
        model: Optional[str] = None,
        ngram_size: Optional[int] = None,
        embedding_method: Optional[str] = None,
    ) -> list[tuple[str, str, float]]:
        """Get similarity pairs for visualization purposes.

        This method provides high-level access to similarity data formatted
        as pairs with full relationship information, suitable for visualization.

        Parameters
        ----------
        article_links : list[str]
            List of article links to get similarities for
        model : Optional[str], default None
            Filter by model if provided
        ngram_size : Optional[int], default None
            Filter by N-gram size if provided
        embedding_method : Optional[str], default None
            Filter by embedding method if provided

        Returns
        -------
        list[tuple[str, str, float]]
            List of tuples containing (article_id_a, article_id_b, similarity_score)
        """
        return self.repository.find_pairs_for_articles(article_links, model, ngram_size, embedding_method)
