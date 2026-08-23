import math
from collections import Counter
from typing import Dict, List, Tuple

from submission.indexer import InvertedIndex, tokenize

_INDEX: InvertedIndex | None = None
_IDF: Dict[str, float] = {}


def build(index: InvertedIndex) -> None:
    global _INDEX, _IDF

    _INDEX = index
    _IDF = {}

    if index.N <= 0:
        return

    for term, posting in index.postings.items():
        df = len(posting)

        _IDF[term] = math.log(
            ((index.N - df + 0.5) / (df + 0.5)) + 1.0
        )


def score(
    query: str,
    k: int,
    k1: float = 1.2,
    b: float = 0.75
) -> List[Tuple[str, float]]:

    if _INDEX is None:
        raise RuntimeError("bm25.build() must be called before score().")

    if k <= 0:
        return []

    if _INDEX.N == 0 or _INDEX.avg_doc_len == 0:
        return []

    terms = tokenize(query)

    if not terms:
        return []

    query_tf = Counter(terms)

    scores: Dict[str, float] = {}

    for term, qtf in query_tf.items():

        posting = _INDEX.postings.get(term)

        if not posting:
            continue

        idf = _IDF.get(term)

        if idf is None:
            df = len(posting)
            idf = math.log(
                ((_INDEX.N - df + 0.5) / (df + 0.5)) + 1.0
            )

        for doc_id, tf in posting.items():

            doc_length = _INDEX.doc_len[doc_id]

            denominator = (
                tf
                + k1
                * (
                    1.0
                    - b
                    + b * (doc_length / _INDEX.avg_doc_len)
                )
            )

            term_score = (
                idf
                * (
                    tf * (k1 + 1.0)
                    / denominator
                )
            )

            scores[doc_id] = (
                scores.get(doc_id, 0.0)
                + qtf * term_score
            )

    results = list(scores.items())

    results.sort(
        key=lambda item: (-item[1], item[0])
    )

    return results[:k]