import math
from collections import Counter
from typing import Dict, List, Tuple

from submission.indexer import InvertedIndex, tokenize

_INDEX: InvertedIndex | None = None
_DOC_NORMS: Dict[str, float] = {}


def build(index: InvertedIndex) -> None:
    global _INDEX, _DOC_NORMS

    _INDEX = index
    _DOC_NORMS = {}

    squared_norms: Dict[str, float] = {}

    for term, posting in index.postings.items():
        df = len(posting)

        if df == 0:
            continue

        idf = math.log(index.N / df) if index.N > 0 else 0.0

        if idf == 0.0:
            continue

        for doc_id, tf in posting.items():
            weight = tf * idf
            squared_norms[doc_id] = squared_norms.get(doc_id, 0.0) + weight * weight

    for doc_id in index.doc_len:
        _DOC_NORMS[doc_id] = math.sqrt(squared_norms.get(doc_id, 0.0))


def boolean_search(query: str, mode: str = "and") -> List[str]:
    if _INDEX is None:
        raise RuntimeError("boolean_vsm.build() must be called before boolean_search().")

    terms = tokenize(query)

    if not terms:
        return []

    posting_sets = []

    for term in terms:
        posting = _INDEX.postings.get(term, {})
        posting_sets.append(set(posting.keys()))

    if mode.lower() == "and":
        if not posting_sets:
            return []

        result = posting_sets[0]

        for docs in posting_sets[1:]:
            result = result.intersection(docs)

    elif mode.lower() == "or":
        result = set()

        for docs in posting_sets:
            result = result.union(docs)

    else:
        raise ValueError("mode must be 'and' or 'or'")

    return sorted(result)


def vsm_score(query: str, k: int) -> List[Tuple[str, float]]:
    if _INDEX is None:
        raise RuntimeError("boolean_vsm.build() must be called before vsm_score().")

    if k <= 0:
        return []

    terms = tokenize(query)

    if not terms:
        return []

    query_tf = Counter(terms)
    query_weights: Dict[str, float] = {}

    query_norm_squared = 0.0

    for term, tf in query_tf.items():
        df = _INDEX.document_frequency(term)

        if df == 0:
            continue

        idf = math.log(_INDEX.N / df) if _INDEX.N > 0 else 0.0

        weight = tf * idf
        query_weights[term] = weight
        query_norm_squared += weight * weight

    query_norm = math.sqrt(query_norm_squared)

    if query_norm == 0.0:
        return []

    dot_products: Dict[str, float] = {}

    for term, query_weight in query_weights.items():
        posting = _INDEX.postings.get(term, {})
        df = len(posting)

        if df == 0:
            continue

        idf = math.log(_INDEX.N / df)

        for doc_id, tf in posting.items():
            document_weight = tf * idf
            dot_products[doc_id] = (
                dot_products.get(doc_id, 0.0)
                + query_weight * document_weight
            )

    results: List[Tuple[str, float]] = []

    for doc_id, dot_product in dot_products.items():
        document_norm = _DOC_NORMS.get(doc_id, 0.0)

        if document_norm == 0.0:
            continue

        score = dot_product / (query_norm * document_norm)

        results.append((doc_id, score))

    results.sort(key=lambda item: (-item[1], item[0]))

    return results[:k]