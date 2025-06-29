from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class ArticleModel(Base):
    """SQLAlchemy model for storing article data.

    Attributes
    ----------
    id : str
        Unique identifier for the article (primary key).
    creator : str
        Creator/author of the article.
    pub_date : str
        Publication date of the article.
    post_date : str
        Post date of the article.
    title : str
        Title of the article.
    body_md : str
        Article body in Markdown format.
    body_html : str
        Article body in HTML format.
    post_id : int
        Post ID from the original source.
    post_type : str
        Type of the post.
    status : str
        Status of the article.
    """

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
    """SQLAlchemy model for storing POS n-gram similarity results.

    Attributes
    ----------
    id : int
        Auto-incrementing primary key.
    article_id_a : str
        ID of the first article in the comparison.
    article_id_b : str
        ID of the second article in the comparison.
    model : str
        Name of the model used for similarity calculation.
    ngram_size : int
        Size of the n-grams used in the calculation.
    embedding_method : str
        Method used for generating embeddings.
    ngram_similarity : float
        Calculated similarity score between the articles.
    """

    __tablename__ = "pos_ngram_similarity"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id_a: Mapped[str] = mapped_column(String)
    article_id_b: Mapped[str] = mapped_column(String)
    model: Mapped[str] = mapped_column(String)
    ngram_size: Mapped[int] = mapped_column(Integer)
    embedding_method: Mapped[str] = mapped_column(String)
    ngram_similarity: Mapped[float] = mapped_column(Float)
