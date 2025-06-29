from typing import List
from datetime import datetime

from src.database import DatabaseManager
from src.models import ArticleModel
from src.entity.article import Article, Content


class ArticleService:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def save_article(self, article: Article) -> None:
        with self.db_manager.get_session() as session:
            article_model = ArticleModel(
                id=article.link,
                creator=article.creator,
                pub_date=article.pub_date.isoformat() if article.pub_date else "",
                post_date=article.post_date.isoformat() if article.post_date else "",
                title=article.title,
                body_md=article.content.markdown,
                body_html=article.content.html,
                post_id=article.post_id,
                post_type=article.post_type,
                status=article.status
            )
            
            existing = session.get(ArticleModel, article.link)
            if existing:
                existing.creator = article_model.creator
                existing.pub_date = article_model.pub_date
                existing.post_date = article_model.post_date
                existing.title = article_model.title
                existing.body_md = article_model.body_md
                existing.body_html = article_model.body_html
                existing.post_id = article_model.post_id
                existing.post_type = article_model.post_type
                existing.status = article_model.status
            else:
                session.add(article_model)
    
    def save_articles(self, articles: List[Article]) -> None:
        with self.db_manager.get_session() as session:
            # 既存の記事IDを取得
            existing_ids = set(session.query(ArticleModel.id).all())
            existing_ids = {article_id[0] for article_id in existing_ids}
            
            # 新規記事と更新記事を分別
            new_articles = []
            update_articles = []
            
            for article in articles:
                article_data = {
                    'id': article.link,
                    'creator': article.creator,
                    'pub_date': article.pub_date.isoformat() if article.pub_date else "",
                    'post_date': article.post_date.isoformat() if article.post_date else "",
                    'title': article.title,
                    'body_md': article.content.markdown,
                    'body_html': article.content.html,
                    'post_id': article.post_id,
                    'post_type': article.post_type,
                    'status': article.status
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
    
    def get_article_count(self) -> int:
        with self.db_manager.get_session() as session:
            return session.query(ArticleModel).count()
    
    def get_articles_by_creator(self, creator: str, newest_first: bool = True) -> List[Article]:
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
                    status=model.status
                )
                articles.append(article)
            return articles