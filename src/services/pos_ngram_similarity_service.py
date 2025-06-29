from typing import List, Optional

from src.database import DatabaseManager
from src.models import PosNgramSimilarityModel
from src.entity.pos_ngram_similarity import PosNgramSimilarityResult


class PosNgramSimilarityService:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def save_similarity(self, result: PosNgramSimilarityResult) -> None:
        with self.db_manager.get_session() as session:
            similarity_model = PosNgramSimilarityModel(
                article_id_a=result.article_id_a,
                article_id_b=result.article_id_b,
                model=result.model,
                ngram_size=result.ngram_size,
                embedding_method=result.embedding_method,
                ngram_similarity=result.ngram_similarity
            )
            session.add(similarity_model)
    
    def save_similarities(self, results: List[PosNgramSimilarityResult]) -> None:
        with self.db_manager.get_session() as session:
            similarity_data = []
            for result in results:
                similarity_data.append({
                    'article_id_a': result.article_id_a,
                    'article_id_b': result.article_id_b,
                    'model': result.model,
                    'ngram_size': result.ngram_size,
                    'embedding_method': result.embedding_method,
                    'ngram_similarity': result.ngram_similarity
                })
            
            if similarity_data:
                session.bulk_insert_mappings(PosNgramSimilarityModel, similarity_data)
    
    def get_similarity(self, article_link_a: str, article_link_b: str, model: str, 
                      ngram_size: int, embedding_method: str) -> Optional[float]:
        with self.db_manager.get_session() as session:
            result = session.query(PosNgramSimilarityModel).filter(
                ((PosNgramSimilarityModel.article_id_a == article_link_a) & 
                 (PosNgramSimilarityModel.article_id_b == article_link_b)) |
                ((PosNgramSimilarityModel.article_id_a == article_link_b) & 
                 (PosNgramSimilarityModel.article_id_b == article_link_a)),
                PosNgramSimilarityModel.model == model,
                PosNgramSimilarityModel.ngram_size == ngram_size,
                PosNgramSimilarityModel.embedding_method == embedding_method
            ).first()
            
            return result.ngram_similarity if result else None
    
    def similarity_exists(self, article_link_a: str, article_link_b: str, model: str, 
                         ngram_size: int, embedding_method: str) -> bool:
        return self.get_similarity(article_link_a, article_link_b, model, ngram_size, embedding_method) is not None