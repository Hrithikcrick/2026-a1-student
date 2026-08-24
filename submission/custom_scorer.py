import heapq
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

from submission.indexer import InvertedIndex, tokenize

_INDEX: Optional[InvertedIndex] = None
_BM25_IDF: Dict[str, float] = {}
_VSM_IDF: Dict[str, float] = {}
_DOC_NORMS: Dict[str, float] = {}

K1 = 2.4
B = 0.60
CANDIDATE_K = 100

VSM_WEIGHT = 0.225
COVERAGE_WEIGHT = 0.15
RARE_WEIGHT = 0.05


def build(index: InvertedIndex) -> None:
    global _INDEX, _BM25_IDF, _VSM_IDF, _DOC_NORMS

    _INDEX = index
    _BM25_IDF = {}
    _VSM_IDF = {}
    _DOC_NORMS = {}

    if index.N <= 0:
        return

    squared_norms: Dict[str, float] = {}

    for term, posting in index.postings.items():
        df = len(posting)

        if df == 0:
            continue

        bm25_idf = math.log(
            ((index.N - df + 0.5) / (df + 0.5)) + 1.0
        )

        vsm_idf = math.log(index.N / df)

        _BM25_IDF[term] = bm25_idf
        _VSM_IDF[term] = vsm_idf

        if vsm_idf == 0.0:
            continue

        for doc_id, tf in posting.items():
            weight = tf * vsm_idf
            squared_norms[doc_id] = (
                squared_norms.get(doc_id, 0.0)
                + weight * weight
            )

    for doc_id, value in squared_norms.items():
        _DOC_NORMS[doc_id] = math.sqrt(value)


def score(query: str, k: int) -> List[Tuple[str, float]]:
    if _INDEX is None:
        raise RuntimeError("custom_scorer.build() must be called before score().")

    if k <= 0 or _INDEX.N == 0 or _INDEX.avg_doc_len == 0:
        return []

    terms = tokenize(query)

    if not terms:
        return []

    query_tf = Counter(terms)

    bm25_scores: Dict[str, float] = {}
    dot_products: Dict[str, float] = {}
    matched_terms: Dict[str, int] = {}
    matched_idf: Dict[str, float] = {}

    query_norm_squared = 0.0
    total_query_idf = 0.0

    for term, qtf in query_tf.items():
        posting = _INDEX.postings.get(term)

        if not posting:
            continue

        bm25_idf = _BM25_IDF.get(term, 0.0)
        vsm_idf = _VSM_IDF.get(term, 0.0)

        total_query_idf += bm25_idf

        query_weight = qtf * vsm_idf
        query_norm_squared += query_weight * query_weight

        for doc_id, tf in posting.items():
            doc_length = _INDEX.doc_len[doc_id]

            denominator = (
                tf
                + K1
                * (
                    1.0
                    - B
                    + B * (doc_length / _INDEX.avg_doc_len)
                )
            )

            bm25_term_score = (
                bm25_idf
                * tf
                * (K1 + 1.0)
                / denominator
            )

            bm25_scores[doc_id] = (
                bm25_scores.get(doc_id, 0.0)
                + qtf * bm25_term_score
            )

            if vsm_idf != 0.0:
                document_weight = tf * vsm_idf

                dot_products[doc_id] = (
                    dot_products.get(doc_id, 0.0)
                    + query_weight * document_weight
                )

            matched_terms[doc_id] = (
                matched_terms.get(doc_id, 0) + 1
            )

            matched_idf[doc_id] = (
                matched_idf.get(doc_id, 0.0)
                + bm25_idf
            )

    if not bm25_scores:
        return []

    query_norm = math.sqrt(query_norm_squared)

    vsm_scores: Dict[str, float] = {}

    if query_norm > 0.0:
        for doc_id, dot_product in dot_products.items():
            doc_norm = _DOC_NORMS.get(doc_id, 0.0)

            if doc_norm > 0.0:
                vsm_scores[doc_id] = (
                    dot_product
                    / (query_norm * doc_norm)
                )

    bm25_top = heapq.nsmallest(
        CANDIDATE_K,
        bm25_scores.items(),
        key=lambda item: (-item[1], item[0])
    )

    vsm_top = heapq.nsmallest(
        CANDIDATE_K,
        vsm_scores.items(),
        key=lambda item: (-item[1], item[0])
    )

    bm25_top_scores = dict(bm25_top)
    vsm_top_scores = dict(vsm_top)

    candidates = set(bm25_top_scores)
    candidates.update(vsm_top_scores)

    max_bm25 = max(
        bm25_top_scores.values(),
        default=1.0
    )

    if max_bm25 <= 0.0:
        max_bm25 = 1.0

    unique_query_terms = len(query_tf)

    results = []

    for doc_id in candidates:
        normalized_bm25 = (
            bm25_top_scores.get(doc_id, 0.0)
            / max_bm25
        )

        vsm_value = vsm_top_scores.get(doc_id, 0.0)

        coverage = (
            matched_terms.get(doc_id, 0)
            / unique_query_terms
        )

        rare_coverage = (
            matched_idf.get(doc_id, 0.0)
            / total_query_idf
            if total_query_idf > 0.0
            else 0.0
        )

        final_score = (
            normalized_bm25
            + VSM_WEIGHT * vsm_value
            + COVERAGE_WEIGHT * coverage
            + RARE_WEIGHT * rare_coverage
        )

        results.append((doc_id, final_score))

    results.sort(
        key=lambda item: (-item[1], item[0])
    )

    return results[:k]