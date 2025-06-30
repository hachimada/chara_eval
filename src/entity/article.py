from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from markdownify import markdownify as md

if TYPE_CHECKING:
    from src.entity.pos_ngram_similarity import PosNgramSimilarityResult


@dataclass
class Content:
    """A class to represent and handle article content.

    This class stores the original HTML content and provides a
    converted Markdown version upon initialization.

    Attributes
    ----------
    html : str
        The original content in HTML format.
    markdown : str
        The content converted to Markdown format. Images are stripped
        during conversion.
    """

    html: str
    markdown: str = field(init=False, repr=False)  # repr=Falseで、print時に長文が出力されないようにする

    def __post_init__(self):
        """
        Post-initialization processor to convert HTML to Markdown.

        This method is automatically called by the dataclass after the
        object has been initialized. It uses the markdownify library
        to perform the conversion, ignoring 'img' tags.
        """
        if self.html:
            # strip=['img'] オプションで画像タグを無視します
            self.markdown = md(self.html, strip=["img"])
        else:
            self.markdown = ""


@dataclass
class Article:
    """A class to represent a blog article.

    Attributes
    ----------
    title : str
        The title of the article.
    link : str
        The permanent link to the article.
    creator : str
        The author of the article.
    content : Content
        The content of the article, stored as a Content object.
    post_id : int
        The post ID of the article.
    pub_date : datetime
        The publication date of the article.
    post_date : datetime
        The post date of the article.
    post_type : str
        The type of the post (e.g., 'post').
    status : str
        The status of the post (e.g., 'publish').
    similarity : Optional[list[PosNgramSimilarityResult]]
        List of similarity results to other articles. This field is populated
        when similarity analysis is performed and represents the similarity
        of this article to other articles.
    """

    title: str = ""
    link: str = ""
    creator: str = ""
    content: Content = field(default_factory=lambda: Content(html=""))
    post_id: int = 0
    pub_date: datetime = field(default_factory=datetime.now)
    post_date: datetime = field(default_factory=datetime.now)
    post_type: str = ""
    status: str = ""
    similarity: Optional[list[PosNgramSimilarityResult]] = None
