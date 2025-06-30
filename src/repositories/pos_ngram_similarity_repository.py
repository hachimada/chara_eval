"""Repository for POS N-gram similarity database operations."""

from typing import Optional

from src.database import DatabaseManager
from src.entity.pos_ngram_similarity import PosNgramSimilarityResult
from src.models.article import PosNgramSimilarityModel
from src.repositories.base_repository import BaseRepository


class PosNgramSimilarityRepository(BaseRepository[PosNgramSimilarityResult]):
    """Repository for managing POS N-gram similarity database operations.

    This repository handles all database operations related to POS N-gram similarity
    results between articles, including saving, retrieving, and checking existence.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize repository with database manager.

        Parameters
        ----------
        db_manager : DatabaseManager
            Database manager instance for database operations.
        """
        super().__init__(db_manager)

    def save_bulk(self, article_similarities: list[tuple[str, PosNgramSimilarityResult]]) -> None:
        """Save multiple POS N-gram similarity results to the database using bulk insert.

        Parameters
        ----------
        article_similarities : list[tuple[str, PosNgramSimilarityResult]]
            List of tuples containing (article_id, similarity_result)
        """
        with self.db_manager.get_session() as session:
            similarity_data = []
            for article_id, result in article_similarities:
                similarity_data.append(
                    {
                        "article_id_a": article_id,
                        "article_id_b": result.other_article_id,
                        "model": result.model,
                        "ngram_size": result.ngram_size,
                        "embedding_method": result.embedding_method,
                        "ngram_similarity": result.ngram_similarity,
                    }
                )

            if similarity_data:
                session.bulk_insert_mappings(PosNgramSimilarityModel, similarity_data)

    def exists_for_article(
        self, article_id: str, other_article_id: str, model: str, ngram_size: int, embedding_method: str
    ) -> bool:
        """Check if similarity exists from an article to another article with specific parameters.

        Parameters
        ----------
        article_id : str
            ID of the source article
        other_article_id : str
            ID of the target article
        model : str
            Model used for similarity calculation
        ngram_size : int
            Size of N-grams used
        embedding_method : str
            Method used for embedding

        Returns
        -------
        bool
            True if similarity exists, False otherwise
        """
        with self.db_manager.get_session() as session:
            result = (
                session.query(PosNgramSimilarityModel)
                .filter(
                    (
                        (PosNgramSimilarityModel.article_id_a == article_id)
                        & (PosNgramSimilarityModel.article_id_b == other_article_id)
                    )
                    | (
                        (PosNgramSimilarityModel.article_id_a == other_article_id)
                        & (PosNgramSimilarityModel.article_id_b == article_id)
                    ),
                    PosNgramSimilarityModel.model == model,
                    PosNgramSimilarityModel.ngram_size == ngram_size,
                    PosNgramSimilarityModel.embedding_method == embedding_method,
                )
                .first()
            )

            return result is not None

    def find_all(
        self, model: Optional[str] = None, ngram_size: Optional[int] = None, embedding_method: Optional[str] = None
    ) -> list[PosNgramSimilarityResult]:
        """Get all similarity results with optional filtering.

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
        with self.db_manager.get_session() as session:
            query = session.query(PosNgramSimilarityModel)

            if model:
                query = query.filter(PosNgramSimilarityModel.model == model)
            if ngram_size:
                query = query.filter(PosNgramSimilarityModel.ngram_size == ngram_size)
            if embedding_method:
                query = query.filter(PosNgramSimilarityModel.embedding_method == embedding_method)

            results = query.all()

            # Convert database results to entity format (choose article_id_b as "other")
            entity_results = []
            for result in results:
                entity_results.append(
                    PosNgramSimilarityResult(
                        other_article_id=result.article_id_b,
                        model=result.model,
                        ngram_size=result.ngram_size,
                        embedding_method=result.embedding_method,
                        ngram_similarity=result.ngram_similarity,
                    )
                )
            return entity_results

    def find_pairs_for_articles(
        self,
        article_links: list[str],
        model: Optional[str] = None,
        ngram_size: Optional[int] = None,
        embedding_method: Optional[str] = None,
    ) -> list[tuple[str, str, float]]:
        """Get similarity pairs for specific articles with optional filtering.

        Parameters
        ----------
        article_links : list[str]
            List of article links to filter by
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
        with self.db_manager.get_session() as session:
            article_link_set = set(article_links)
            query = session.query(PosNgramSimilarityModel).filter(
                PosNgramSimilarityModel.article_id_a.in_(article_links),
                PosNgramSimilarityModel.article_id_b.in_(article_links),
            )

            if model:
                query = query.filter(PosNgramSimilarityModel.model == model)
            if ngram_size:
                query = query.filter(PosNgramSimilarityModel.ngram_size == ngram_size)
            if embedding_method:
                query = query.filter(PosNgramSimilarityModel.embedding_method == embedding_method)

            results = query.all()

            # Return tuples with full relationship information
            pairs = []
            processed_pairs = set()
            for result in results:
                if result.article_id_a in article_link_set and result.article_id_b in article_link_set:
                    # Create a normalized pair key to avoid duplicates
                    pair_key = tuple(sorted([result.article_id_a, result.article_id_b]))
                    if pair_key not in processed_pairs:
                        pairs.append((result.article_id_a, result.article_id_b, result.ngram_similarity))
                        processed_pairs.add(pair_key)

            return pairs
