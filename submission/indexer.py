import os
import pickle
import re
from typing import Dict, List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_INDEX_FILENAME = "inverted_index.pkl"


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class InvertedIndex:
    def __init__(self):
        self.postings: Dict[str, Dict[str, int]] = {}
        self.doc_len: Dict[str, int] = {}
        self.doc_text: Dict[str, str] = {}
        self.N: int = 0
        self.avg_doc_len: float = 0.0

    def build(self, corpus: List[Tuple[str, str]]) -> None:
        self.postings = {}
        self.doc_len = {}
        self.doc_text = {}
        self.N = 0
        self.avg_doc_len = 0.0

        total_doc_len = 0

        for doc_id, text in corpus:
            tokens = tokenize(text)

            self.N += 1
            self.doc_len[doc_id] = len(tokens)
            self.doc_text[doc_id] = text
            total_doc_len += len(tokens)

            term_freqs: Dict[str, int] = {}

            for term in tokens:
                term_freqs[term] = term_freqs.get(term, 0) + 1

            for term, tf in term_freqs.items():
                if term not in self.postings:
                    self.postings[term] = {}

                self.postings[term][doc_id] = tf

        if self.N > 0:
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
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

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