from datetime import datetime
from sqlalchemy import String, Text, DateTime, Integer, Float
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ArticleModel(Base):
    __tablename__ = "article"
    
    id: Mapped[str] = mapped_column(String, primary_key=True)
    creator: Mapped[str] = mapped_column(String)
    pub_date: Mapped[str] = mapped_column(String)
    post_date: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(String)
    body_md: Mapped[str] = mapped_column(Text)
    body_html: Mapped[str] = mapped_column(Text)
    post_id: Mapped[int] = mapped_column(Integer)
    post_type: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)


class PosNgramSimilarityModel(Base):
    __tablename__ = "pos_ngram_similarity"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id_a: Mapped[str] = mapped_column(String)
    article_id_b: Mapped[str] = mapped_column(String)
    model: Mapped[str] = mapped_column(String)
    ngram_size: Mapped[int] = mapped_column(Integer)
    embedding_method: Mapped[str] = mapped_column(String)
    ngram_similarity: Mapped[float] = mapped_column(Float)