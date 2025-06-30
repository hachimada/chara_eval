"""Repository for POS N-gram similarity database operations."""

from typing import Optional

from sqlalchemy import or_
from tqdm import tqdm

from src.database import DatabaseManager
from src.entity.pos_ngram_similarity import PosNgramSimilarityResult
from src.models import PosNgramSimilarityModel
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

    def save(self, result: PosNgramSimilarityResult) -> None:
        """Save a single POS N-gram similarity result to the database.

        Parameters
        ----------
        result : PosNgramSimilarityResult
            The similarity result to save
        """
        with self.db_manager.get_session() as session:
            similarity_model = PosNgramSimilarityModel(
                article_id_a=result.article_id_a,
                article_id_b=result.article_id_b,
                model=result.model,
                ngram_size=result.ngram_size,
                embedding_method=result.embedding_method,
                ngram_similarity=result.ngram_similarity,
            )
            session.add(similarity_model)

    def save_bulk(self, results: list[PosNgramSimilarityResult]) -> None:
        """Save multiple POS N-gram similarity results to the database using bulk insert.

        Parameters
        ----------
        results : list[PosNgramSimilarityResult]
            List of similarity results to save
        """
        with self.db_manager.get_session() as session:
            similarity_data = []
            for result in results:
                similarity_data.append(
                    {
                        "article_id_a": result.article_id_a,
                        "article_id_b": result.article_id_b,
                        "model": result.model,
                        "ngram_size": result.ngram_size,
                        "embedding_method": result.embedding_method,
                        "ngram_similarity": result.ngram_similarity,
                    }
                )

            if similarity_data:
                session.bulk_insert_mappings(PosNgramSimilarityModel, similarity_data)

    def find_similarity(
        self, article_link_a: str, article_link_b: str, model: str, ngram_size: int, embedding_method: str
    ) -> Optional[float]:
        """Get similarity value between two articles with specific parameters.

        Parameters
        ----------
        article_link_a : str
            ID of the first article
        article_link_b : str
            ID of the second article
        model : str
            Model used for similarity calculation
        ngram_size : int
            Size of N-grams used
        embedding_method : str
            Method used for embedding

        Returns
        -------
        Optional[float]
            Similarity value if found, None otherwise
        """
        with self.db_manager.get_session() as session:
            result = (
                session.query(PosNgramSimilarityModel)
                .filter(
                    (
                        (PosNgramSimilarityModel.article_id_a == article_link_a)
                        & (PosNgramSimilarityModel.article_id_b == article_link_b)
                    )
                    | (
                        (PosNgramSimilarityModel.article_id_a == article_link_b)
                        & (PosNgramSimilarityModel.article_id_b == article_link_a)
                    ),
                    PosNgramSimilarityModel.model == model,
                    PosNgramSimilarityModel.ngram_size == ngram_size,
                    PosNgramSimilarityModel.embedding_method == embedding_method,
                )
                .first()
            )

            return result.ngram_similarity if result else None

    def exists(
        self, article_link_a: str, article_link_b: str, model: str, ngram_size: int, embedding_method: str
    ) -> bool:
        """Check if similarity exists between two articles with specific parameters.

        Parameters
        ----------
        article_link_a : str
            ID of the first article
        article_link_b : str
            ID of the second article
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
        return self.find_similarity(article_link_a, article_link_b, model, ngram_size, embedding_method) is not None

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

            return [
                PosNgramSimilarityResult(
                    article_id_a=result.article_id_a,
                    article_id_b=result.article_id_b,
                    model=result.model,
                    ngram_size=result.ngram_size,
                    embedding_method=result.embedding_method,
                    ngram_similarity=result.ngram_similarity,
                )
                for result in results
            ]

    def find_by_pairs(
        self, pairs: list[tuple[str, str]], model: str, ngram_size: int, embedding_method: str
    ) -> list[PosNgramSimilarityResult]:
        """Get similarities for specific article pairs with given parameters.

        Parameters
        ----------
        pairs : list[tuple[str, str]]
            List of article ID pairs to retrieve similarities for
        model : str
            Model used for similarity calculation
        ngram_size : int
            Size of N-grams used
        embedding_method : str
            Method used for embedding

        Returns
        -------
        list[PosNgramSimilarityResult]
            List of similarity results for the specified pairs
        """
        with self.db_manager.get_session() as session:
            all_results = []
            chunk_size = 100
            pb = tqdm(total=len(pairs), desc="Fetching similarities by pairs")
            for i in range(0, len(pairs), chunk_size):
                chunk = pairs[i : i + chunk_size]
                conditions = []
                for a, b in chunk:
                    conditions.append(
                        ((PosNgramSimilarityModel.article_id_a == a) & (PosNgramSimilarityModel.article_id_b == b))
                        | ((PosNgramSimilarityModel.article_id_a == b) & (PosNgramSimilarityModel.article_id_b == a))
                    )

                if not conditions:
                    continue

                results = (
                    session.query(PosNgramSimilarityModel)
                    .filter(
                        or_(*conditions),
                        PosNgramSimilarityModel.model == model,
                        PosNgramSimilarityModel.ngram_size == ngram_size,
                        PosNgramSimilarityModel.embedding_method == embedding_method,
                    )
                    .all()
                )

                all_results.extend(results)
                pb.update(len(chunk))

            return [
                PosNgramSimilarityResult(
                    article_id_a=result.article_id_a,
                    article_id_b=result.article_id_b,
                    model=result.model,
                    ngram_size=result.ngram_size,
                    embedding_method=result.embedding_method,
                    ngram_similarity=result.ngram_similarity,
                )
                for result in all_results
            ]
