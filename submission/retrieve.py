from typing import List, Optional, Tuple

from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex
from submission import custom_scorer

_INDEX: Optional[InvertedIndex] = None


def build_index(corpus_path: str, index_dir: str) -> None:
    corpus = load_corpus(corpus_path)

    index = InvertedIndex()
    index.build(corpus)
    index.save(index_dir)


def load_index(index_dir: str) -> None:
    global _INDEX

    _INDEX = InvertedIndex.load(index_dir)
    custom_scorer.build(_INDEX)


def retrieve(query: str, k: int = 10) -> List[Tuple[str, float]]:
    if _INDEX is None:
        raise RuntimeError("load_index() must be called before retrieve().")

    return custom_scorer.score(query, k)