import os
import pickle
import re
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Tuple

from nltk.stem import PorterStemmer

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_INDEX_FILENAME = "inverted_index.pkl"
_STEMMER = PorterStemmer()

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being",
    "but", "by", "can", "could", "did", "do", "does", "doing",
    "for", "from", "had", "has", "have", "having", "he", "her",
    "hers", "him", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "may", "might", "more", "most", "my",
    "no", "not", "of", "on", "or", "our", "ours", "out", "over",
    "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "under",
    "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "why", "will", "with", "would", "you",
    "your", "yours"
}


@lru_cache(maxsize=200000)
def _stem(token: str) -> str:
    return _STEMMER.stem(token)


def tokenize(text: str) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    stopwords = _STOPWORDS
    stem = _stem
    return [
        stem(token)
        for token in tokens
        if token not in stopwords
    ]


class InvertedIndex:
    def __init__(self):
        self.postings: Dict[str, Dict[str, int]] = {}
        self.doc_len: Dict[str, int] = {}
        self.doc_text: Dict[str, str] = {}
        self.N: int = 0
        self.avg_doc_len: float = 0.0

    def build(self, corpus: List[Tuple[str, str]]) -> None:
        postings: Dict[str, Dict[str, int]] = {}
        doc_len: Dict[str, int] = {}

        total_doc_len = 0

        for doc_id, text in corpus:
            tokens = tokenize(text)
            length = len(tokens)

            doc_len[doc_id] = length
            total_doc_len += length

            for term, tf in Counter(tokens).items():
                posting = postings.get(term)

                if posting is None:
                    postings[term] = {doc_id: tf}
                else:
                    posting[doc_id] = tf

        self.postings = postings
        self.doc_len = doc_len
        self.doc_text = {}
        self.N = len(corpus)

        if self.N:
            self.avg_doc_len = total_doc_len / self.N
        else:
            self.avg_doc_len = 0.0

    def document_frequency(self, term: str) -> int:
        return len(self.postings.get(term, {}))

    def save(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)

        state = {
            "postings": self.postings,
            "doc_len": self.doc_len,
            "N": self.N,
            "avg_doc_len": self.avg_doc_len,
        }

        path = os.path.join(index_dir, _INDEX_FILENAME)

        with open(path, "wb") as f:
            pickle.dump(
                state,
                f,
                protocol=pickle.HIGHEST_PROTOCOL
            )

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        path = os.path.join(index_dir, _INDEX_FILENAME)

        with open(path, "rb") as f:
            state = pickle.load(f)

        index = cls()

        index.postings = state["postings"]
        index.doc_len = state["doc_len"]
        index.N = state["N"]
        index.avg_doc_len = state["avg_doc_len"]
        index.doc_text = {}

        return index