"""
Core Similarity Logic for Recommendation Systems
=================================================
This module provides first-principles implementations of similarity metrics
used in modern AI recommendation engines:
1. Cosine Similarity (direction/orientation in vector space)
2. Jaccard Similarity (set-overlap for discrete tags / categories)
3. Euclidean Distance & Similarity (spatial geometric closeness)
4. Dot Product (projection, base of neural & matrix factorization scoring)
5. TF-IDF Vectorizer (text-to-vector representation for unstructured descriptions)
"""

from __future__ import annotations
import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Set, Tuple


def dot_product(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """
    Computes standard dot product: sum(a_i * b_i).
    Used in latent factor models, embeddings, and matrix factorization.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimensions must match ({len(vec_a)} != {len(vec_b)})")
    return sum(a * b for a, b in zip(vec_a, vec_b))


def vector_norm(vec: Sequence[float]) -> float:
    """Computes L2 norm (magnitude / Euclidean length): sqrt(sum(x_i^2))."""
    return math.sqrt(sum(x * x for x in vec))


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """
    Cosine Similarity:
        cos(theta) = (A . B) / (||A|| * ||B||)

    Properties:
    - Measures angle/direction between vectors, not magnitude.
    - Range: [-1.0, 1.0] for general vectors; [0.0, 1.0] for non-negative feature counts.
    - Key intuition: If two products talk about "noise-canceling bluetooth headphones",
      their term vectors point in roughly the same direction even if one description is
      twice as long as the other.
    """
    norm_a = vector_norm(vec_a)
    norm_b = vector_norm(vec_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product(vec_a, vec_b) / (norm_a * norm_b)


def jaccard_similarity(set_a: Iterable[str], set_b: Iterable[str]) -> float:
    """
    Jaccard Similarity:
        J(A, B) = |A ∩ B| / |A ∪ B|

    Properties:
    - Excellent for discrete categorical attributes, e.g. tags, genres, brand overlap.
    - Range: [0.0, 1.0].
    - Intuition: Ratio of common traits over all distinct traits between two items.
    """
    s_a = set(set_a)
    s_b = set(set_b)

    if not s_a and not s_b:
        return 1.0
    if not s_a or not s_b:
        return 0.0

    intersection = len(s_a.intersection(s_b))
    union = len(s_a.union(s_b))
    return intersection / union if union > 0 else 0.0


def euclidean_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """
    Euclidean Distance:
        d = sqrt(sum((a_i - b_i)^2))
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vector dimensions must match")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))


def euclidean_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """
    Converts Euclidean Distance into a normalized [0, 1] similarity score:
        sim = 1 / (1 + distance)
    """
    dist = euclidean_distance(vec_a, vec_b)
    return 1.0 / (1.0 + dist)


class TFIDFVectorizer:
    """
    A transparent, dependency-free TF-IDF (Term Frequency - Inverse Document Frequency)
    vectorizer.

    Formulas:
        TF(term, doc) = count(term in doc) / total_words_in_doc
        IDF(term)     = ln((1 + N) / (1 + DF(term))) + 1   [smooth IDF formula]
        Weight        = TF * IDF
        Normalized    = Weight / L2_Norm
    """

    def __init__(self, stop_words: Set[str] | None = None):
        self.stop_words = stop_words or {
            "a", "about", "above", "after", "again", "all", "am", "an", "and", "any",
            "are", "as", "at", "be", "because", "been", "before", "being", "below",
            "between", "both", "but", "by", "for", "from", "further", "had", "has",
            "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
            "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me",
            "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on",
            "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
            "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
            "their", "theirs", "them", "themselves", "then", "there", "these", "they",
            "this", "those", "through", "to", "too", "under", "until", "up", "very",
            "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
            "why", "with", "you", "your", "yours", "yourself", "yourselves"
        }
        self.vocabulary: Dict[str, int] = {}
        self.idf: List[float] = []
        self.n_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Cleans and extracts alphanumeric lowercase tokens."""
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]

    def fit(self, documents: Sequence[str]) -> "TFIDFVectorizer":
        """Builds vocabulary and calculates IDF for each term across the corpus."""
        self.n_docs = len(documents)
        doc_tokens = [self._tokenize(doc) for doc in documents]

        # Calculate document frequency: in how many docs does each term appear?
        df: Counter[str] = Counter()
        for tokens in doc_tokens:
            unique_terms = set(tokens)
            for term in unique_terms:
                df[term] += 1

        # Sort terms alphabetically for deterministic index mapping
        sorted_terms = sorted(df.keys())
        self.vocabulary = {term: idx for idx, term in enumerate(sorted_terms)}

        # Smooth IDF calculation: ln((1 + N) / (1 + df)) + 1
        self.idf = [0.0] * len(self.vocabulary)
        for term, idx in self.vocabulary.items():
            doc_freq = df[term]
            self.idf[idx] = math.log((1.0 + self.n_docs) / (1.0 + doc_freq)) + 1.0

        return self

    def transform(self, documents: Sequence[str]) -> List[List[float]]:
        """Transforms documents into L2-normalized TF-IDF feature vectors."""
        if not self.vocabulary:
            raise RuntimeError("Vectorizer must be fitted before calling transform.")

        vectors: List[List[float]] = []
        vocab_size = len(self.vocabulary)

        for doc in documents:
            tokens = self._tokenize(doc)
            if not tokens:
                vectors.append([0.0] * vocab_size)
                continue

            term_counts = Counter(tokens)
            total_tokens = len(tokens)
            vec = [0.0] * vocab_size

            for term, count in term_counts.items():
                if term in self.vocabulary:
                    idx = self.vocabulary[term]
                    tf = count / total_tokens
                    vec[idx] = tf * self.idf[idx]

            # L2 Normalization (ensures dot_product equals cosine similarity)
            norm = vector_norm(vec)
            if norm > 0.0:
                vec = [val / norm for val in vec]

            vectors.append(vec)

        return vectors

    def fit_transform(self, documents: Sequence[str]) -> List[List[float]]:
        return self.fit(documents).transform(documents)

