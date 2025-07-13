import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class CalculationConfig:
    """Configuration for article similarity calculation.

    Attributes
    ----------
    creator : str
        Creator name for filtering articles
    model : str
        SpaCy model name for POS tagging
    ngram_size : int
        Size of N-grams for similarity calculation
    embedding_method : str
        Embedding method ('bow' or 'tfidf')
    xml_file : Path
        Path to input XML file
    total_articles_found : int
        Total number of articles found before filtering
    articles_after_filtering : int
        Number of articles after applying filters
    minimum_content_length : int
        Minimum content length for article filtering
    execution_timestamp : str
        Timestamp when calculation was executed
    """

    creator: str
    model: str
    ngram_size: int
    embedding_method: str
    xml_file: Path
    total_articles_found: int
    articles_after_filtering: int
    minimum_content_length: int = 100
    execution_timestamp: str = None

    def __post_init__(self) -> None:
        """Set execution timestamp if not provided."""
        if self.execution_timestamp is None:
            self.execution_timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Convert configuration to dictionary format.

        Returns
        -------
        dict
            Configuration as dictionary
        """
        return {
            "execution_timestamp": self.execution_timestamp,
            "parameters": {
                "creator": self.creator,
                "model": self.model,
                "ngram_size": self.ngram_size,
                "embedding_method": self.embedding_method,
                "xml_file": str(self.xml_file),
            },
            "data_info": {
                "total_articles_found": self.total_articles_found,
                "articles_after_filtering": self.articles_after_filtering,
                "minimum_content_length": self.minimum_content_length,
            },
        }

    def save_to_json(self, output_dir: Path) -> Path:
        """Save configuration to JSON file.

        Parameters
        ----------
        output_dir : Path
            Directory to save the configuration file

        Returns
        -------
        Path
            Path to the saved configuration file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        config_path = output_dir / "calculation_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

        return config_path

    @classmethod
    def from_json(cls, json_path: Path) -> "CalculationConfig":
        """Load configuration from JSON file.

        Parameters
        ----------
        json_path : Path
            Path to JSON configuration file

        Returns
        -------
        CalculationConfig
            Loaded configuration object
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        params = data["parameters"]
        data_info = data["data_info"]

        return cls(
            creator=params["creator"],
            model=params["model"],
            ngram_size=params["ngram_size"],
            embedding_method=params["embedding_method"],
            xml_file=Path(params["xml_file"]),
            total_articles_found=data_info["total_articles_found"],
            articles_after_filtering=data_info["articles_after_filtering"],
            minimum_content_length=data_info["minimum_content_length"],
            execution_timestamp=data["execution_timestamp"],
        )


@dataclass
class PosNgramSimilarityResult:
    """Result of POS n-gram similarity calculation from one article to another.

    This class represents the similarity relationship from the perspective of
    one article to another article, simplifying the bidirectional relationship
    into a unidirectional one.

    Attributes
    ----------
    other_article_id : str
        ID of the other article being compared to.
    model : str
        Name of the model used for similarity calculation.
    ngram_size : int
        Size of the n-grams used.
    embedding_method : str
        Method used for embedding generation.
    ngram_similarity : float
        Calculated similarity score to the other article.
    """

    other_article_id: str
    model: str
    ngram_size: int
    embedding_method: str
    ngram_similarity: float
