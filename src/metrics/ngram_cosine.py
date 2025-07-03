"""ngram_cosine.py

Utilities for computing cosine similarity between the POS‑N‑gram
frequency distributions of two texts.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List

import spacy

try:
    spacy.load("ja_core_news_sm")  # ここでエラーが出なければ成功
    spacy.load("ja_core_news_md")
except OSError:
    raise ImportError(
        "SpaCy Japanese model 'ja_core_news_sm' is not installed. "
        "Please install it with 'python -m spacy download ja_core_news_sm'."
    )


def _pos_ngrams(doc_tokens: List[str], n: int) -> Counter:
    """Return frequency counter of POS n‑grams for a given list of tokens.

    Parameters
    ----------
    doc_tokens : list of str
        POS tag sequence extracted from a SpaCy `Doc` object.
    n : int
        The *n* in n‑gram size.

    Returns
    -------
    Counter
        Mapping of POS n‑gram tuples to counts.
    """
    return Counter(tuple(doc_tokens[i : i + n]) for i in range(len(doc_tokens) - n + 1))


def embedding(
    counter_a: dict[tuple[str, str], int],
    counter_b: dict[tuple[str, str], int],
    embedding_type: str = "bow",
) -> tuple[list[int], list[int]]:
    """Return the embedding of two documents.

    Parameters
    ----------
    counter_a : dict[tuple[str, str], int]
        - Frequency counter of POS n‑grams for the first document.
        - Example(n=2): {('NOUN', 'VERB'): 3, ('ADJ', 'NOUN'): 2}
    counter_b : dict[tuple[str, str], int]
        - Frequency counter of POS n‑grams for the second document.
        - Example(n=3): {('NOUN', 'VERB', 'ADJ'): 1, ('VERB', 'NOUN', 'ADJ'): 4}
    embedding_type : str, optional
        - Type of embedding to use. Currently supports "bow" (Bag-of-Words).

    Returns
    -------
    tuple[list[int], list[int]]
        - Two lists representing the embeddings of the documents.
    """
    if embedding_type == "bow":
        # Bag-of-Words (BoW) representation
        keys = set(counter_a) | set(counter_b)
        vector_a = [counter_a.get(k, 0) for k in keys]
        vector_b = [counter_b.get(k, 0) for k in keys]
        return vector_a, vector_b
    elif embedding_type == "tfidf":
        # TF-IDF representation (not implemented here)
        raise NotImplementedError("TF-IDF embedding is not implemented.")
    else:
        raise ValueError("Unknown embedding type: {}".format(type))

def morphological_analysis(nlp, text: str) -> tuple[List[str], List[str], List[str]]:
    """Perform morphological analysis on the input text using SpaCy.

    Parameters
    ----------
    nlp : spacy.language.Language
        Loaded SpaCy model for processing.
    text : str
        Input text to analyze.

    Returns
    -------
    tuple[List[str], List[str], List[str]]
        - List of tokens in the text.
        - List of POS tags corresponding to each token.
        - List of lemmas corresponding to each token.
    """
    doc = nlp(text)
    test_list = []
    pos_list = []
    lemma_list = []
    for token in doc:
        test_list.append(token.text)
        pos_list.append(token.pos_)
        lemma_list.append(token.lemma_)
    return test_list, pos_list, lemma_list


def pos_ngram_cosine_similarity(
    text_a: str,
    text_b: str,
    n: int = 2,
    spacy_model: str = "ja_core_news_sm",
    embedding_type: str = "bow",
) -> float:
    """Compute cosine similarity between two texts based on POS n‑grams.

    Parameters
    ----------
    text_a : str
        First input text.
    text_b : str
        Second input text.
    n : int, optional
        Size of the n‑gram window (default is 2).
    spacy_model : str, optional
        Name of the SpaCy model to load (default ``"ja_core_news_sm"``).
    embedding_type : str, optional
        Type of embedding to use for the n‑grams (default is "bow" for Bag-of-Words).

    Returns
    -------
    float
        Cosine similarity in the range ``[0.0, 1.0]``. When either text
        lacks sufficient tokens to form n‑grams, the function returns 0.0.

    Notes
    -----
    The function loads the specified SpaCy model on first call and caches
    it for reuse. POS tags are taken from ``token.pos_``.  Cosine similarity
    is defined as

    .. math::

        \\text{sim}(\\mathbf{A}, \\mathbf{B}) =
        \\frac{\\sum_i A_i B_i}{\\sqrt{\\sum_i A_i^2}\\;\\sqrt{\\sum_i B_i^2}}.

    Examples
    --------
    >>> pos_ngram_cosine_similarity("私は猫です。", "僕は犬だ。")
    0.83  # depending on the tokenizer/pos‑tagger
    """
    # SpaCyモデルは一度読み込んだらキャッシュして再利用する
    if not hasattr(pos_ngram_cosine_similarity, "_nlp"):
        pos_ngram_cosine_similarity._nlp = spacy.load(spacy_model)

    nlp = pos_ngram_cosine_similarity._nlp

    # 形態素解析
    doc_a, pos_a, lemma_a = morphological_analysis(nlp, text_a)
    doc_b, pos_b, lemma_b = morphological_analysis(nlp, text_b)

    # 十分な長さがなければ類似度は0とする
    if len(pos_a) < n or len(pos_b) < n:
        return 0.0

    counter_a = _pos_ngrams(pos_a, n)
    counter_b = _pos_ngrams(pos_b, n)

    # 品詞n-gramのベクトル化
    vector_a, vector_b = embedding(counter_a, counter_b, embedding_type=embedding_type)

    # コサイン類似度の計算
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0  # Avoid division by zero
    return dot_product / (norm_a * norm_b)
