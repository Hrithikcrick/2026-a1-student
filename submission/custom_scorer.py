import heapq
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

from submission.indexer import InvertedIndex, tokenize


_INDEX: Optional[InvertedIndex] = None

_BM25_IDF: Dict[str, float] = {}
_VSM_IDF: Dict[str, float] = {}
_PREFIX_IDF: Dict[str, float] = {}

_DOC_NORMS: Dict[str, float] = {}
_BM25_LENGTH_NORM: Dict[str, float] = {}


K1 = 2.4
B = 0.60

CANDIDATE_K = 100

VSM_WEIGHT = 0.225
COVERAGE_WEIGHT = 0.15
RARE_WEIGHT = 0.05

PREFIX_WEIGHT = 0.08
PREFIX_COORD_WEIGHT = 0.08

PREFIX_BM25_WEIGHT = 0.10
PREFIX_K1 = 1.2


def build(index: InvertedIndex) -> None:

    global _INDEX
    global _BM25_IDF
    global _VSM_IDF
    global _PREFIX_IDF
    global _DOC_NORMS
    global _BM25_LENGTH_NORM

    _INDEX = index

    _BM25_IDF = {}
    _VSM_IDF = {}
    _PREFIX_IDF = {}

    _DOC_NORMS = {}
    _BM25_LENGTH_NORM = {}

    if index.N <= 0:
        return

    N = index.N
    avg_doc_len = index.avg_doc_len

    if avg_doc_len > 0.0:

        for doc_id, doc_length in index.doc_len.items():

            _BM25_LENGTH_NORM[doc_id] = (
                K1
                * (
                    1.0
                    - B
                    + B
                    * (
                        doc_length
                        / avg_doc_len
                    )
                )
            )

    squared_norms: Dict[str, float] = {}

    for term, posting in index.postings.items():

        df = len(posting)

        if df == 0:
            continue

        bm25_idf = math.log(
            (
                (
                    N
                    - df
                    + 0.5
                )
                / (
                    df
                    + 0.5
                )
            )
            + 1.0
        )

        vsm_idf = math.log(
            N / df
        )

        _BM25_IDF[term] = bm25_idf
        _VSM_IDF[term] = vsm_idf

        if vsm_idf == 0.0:
            continue

        for doc_id, tf in posting.items():

            weight = (
                tf
                * vsm_idf
            )

            squared_norms[doc_id] = (
                squared_norms.get(
                    doc_id,
                    0.0
                )
                + weight
                * weight
            )

    for doc_id, value in squared_norms.items():

        _DOC_NORMS[doc_id] = (
            math.sqrt(value)
        )

    prefix_postings = getattr(
        index,
        "prefix_postings",
        {}
    )

    for term, posting in prefix_postings.items():

        df = len(posting)

        if df == 0:
            continue

        prefix_idf = math.log(
            (
                (
                    N
                    - df
                    + 0.5
                )
                / (
                    df
                    + 0.5
                )
            )
            + 1.0
        )

        _PREFIX_IDF[term] = (
            prefix_idf
        )


def score(
    query: str,
    k: int
) -> List[Tuple[str, float]]:

    index = _INDEX

    if index is None:

        raise RuntimeError(
            "custom_scorer.build() must be called before score()."
        )

    if (
        k <= 0
        or index.N == 0
        or index.avg_doc_len == 0
    ):

        return []

    terms = tokenize(query)

    if not terms:
        return []

    query_tf = Counter(terms)

    postings = index.postings
    prefix_postings = getattr(
        index,
        "prefix_postings",
        {}
    )

    bm25_idfs = _BM25_IDF
    vsm_idfs = _VSM_IDF
    prefix_idfs = _PREFIX_IDF

    doc_norms = _DOC_NORMS
    length_norms = _BM25_LENGTH_NORM

    bm25_scores: Dict[str, float] = {}
    dot_products: Dict[str, float] = {}

    matched_terms: Dict[str, int] = {}
    matched_idf: Dict[str, float] = {}

    query_norm_squared = 0.0
    total_query_idf = 0.0

    k1_plus_one = (
        K1 + 1.0
    )

    for term, qtf in query_tf.items():

        posting = postings.get(term)

        if not posting:
            continue

        bm25_idf = bm25_idfs.get(
            term,
            0.0
        )

        vsm_idf = vsm_idfs.get(
            term,
            0.0
        )

        total_query_idf += (
            bm25_idf
        )

        query_weight = (
            qtf
            * vsm_idf
        )

        query_norm_squared += (
            query_weight
            * query_weight
        )

        for doc_id, tf in posting.items():

            denominator = (
                tf
                + length_norms[
                    doc_id
                ]
            )

            bm25_term_score = (
                bm25_idf
                * tf
                * k1_plus_one
                / denominator
            )

            bm25_scores[doc_id] = (
                bm25_scores.get(
                    doc_id,
                    0.0
                )
                + qtf
                * bm25_term_score
            )

            if vsm_idf != 0.0:

                document_weight = (
                    tf
                    * vsm_idf
                )

                dot_products[doc_id] = (
                    dot_products.get(
                        doc_id,
                        0.0
                    )
                    + query_weight
                    * document_weight
                )

            matched_terms[doc_id] = (
                matched_terms.get(
                    doc_id,
                    0
                )
                + 1
            )

            matched_idf[doc_id] = (
                matched_idf.get(
                    doc_id,
                    0.0
                )
                + bm25_idf
            )

    if not bm25_scores:
        return []

    query_norm = math.sqrt(
        query_norm_squared
    )

    vsm_scores: Dict[str, float] = {}

    if query_norm > 0.0:

        inverse_query_norm = (
            1.0
            / query_norm
        )

        for doc_id, dot_product in dot_products.items():

            doc_norm = doc_norms.get(
                doc_id,
                0.0
            )

            if doc_norm > 0.0:

                vsm_scores[doc_id] = (
                    dot_product
                    * inverse_query_norm
                    / doc_norm
                )

    bm25_top = heapq.nsmallest(
        CANDIDATE_K,
        bm25_scores.items(),
        key=lambda item: (
            -item[1],
            item[0]
        )
    )

    vsm_top = heapq.nsmallest(
        CANDIDATE_K,
        vsm_scores.items(),
        key=lambda item: (
            -item[1],
            item[0]
        )
    )

    bm25_top_scores = dict(
        bm25_top
    )

    vsm_top_scores = dict(
        vsm_top
    )

    candidates = set(
        bm25_top_scores
    )

    candidates.update(
        vsm_top_scores
    )

    max_bm25 = max(
        bm25_top_scores.values(),
        default=1.0
    )

    if max_bm25 <= 0.0:
        max_bm25 = 1.0

    inverse_max_bm25 = (
        1.0
        / max_bm25
    )

    unique_query_terms = len(
        query_tf
    )

    inverse_unique_terms = (
        1.0
        / unique_query_terms
    )

    if total_query_idf > 0.0:

        inverse_total_query_idf = (
            1.0
            / total_query_idf
        )

    else:

        inverse_total_query_idf = 0.0

    prefix_matched_terms: Dict[
        str,
        int
    ] = {}

    prefix_matched_idf: Dict[
        str,
        float
    ] = {}

    prefix_bm25_scores: Dict[
        str,
        float
    ] = {}

    total_prefix_query_idf = 0.0

    prefix_k1_plus_one = (
        PREFIX_K1 + 1.0
    )

    for term, qtf in query_tf.items():

        prefix_posting = (
            prefix_postings.get(
                term
            )
        )

        if not prefix_posting:
            continue

        prefix_idf = (
            prefix_idfs.get(
                term,
                0.0
            )
        )

        full_idf = (
            bm25_idfs.get(
                term,
                0.0
            )
        )

        if prefix_idf <= 0.0:
            continue

        total_prefix_query_idf += (
            prefix_idf
        )

        if len(prefix_posting) < len(candidates):

            iterator = (
                (
                    doc_id,
                    tf
                )
                for doc_id, tf
                in prefix_posting.items()
                if doc_id in candidates
            )

        else:

            iterator = (
                (
                    doc_id,
                    prefix_posting[
                        doc_id
                    ]
                )
                for doc_id
                in candidates
                if doc_id
                in prefix_posting
            )

        for doc_id, tf in iterator:

            prefix_matched_terms[
                doc_id
            ] = (
                prefix_matched_terms.get(
                    doc_id,
                    0
                )
                + 1
            )

            prefix_matched_idf[
                doc_id
            ] = (
                prefix_matched_idf.get(
                    doc_id,
                    0.0
                )
                + full_idf
            )

            tf_component = (
                tf
                * prefix_k1_plus_one
                / (
                    tf
                    + PREFIX_K1
                )
            )

            prefix_bm25_scores[
                doc_id
            ] = (
                prefix_bm25_scores.get(
                    doc_id,
                    0.0
                )
                + qtf
                * prefix_idf
                * tf_component
            )

    max_prefix_bm25 = max(
        prefix_bm25_scores.values(),
        default=1.0
    )

    if max_prefix_bm25 <= 0.0:

        max_prefix_bm25 = 1.0

    inverse_max_prefix_bm25 = (
        1.0
        / max_prefix_bm25
    )

    results = []

    for doc_id in candidates:

        normalized_bm25 = (
            bm25_top_scores.get(
                doc_id,
                0.0
            )
            * inverse_max_bm25
        )

        vsm_value = (
            vsm_top_scores.get(
                doc_id,
                0.0
            )
        )

        coverage = (
            matched_terms.get(
                doc_id,
                0
            )
            * inverse_unique_terms
        )

        if inverse_total_query_idf > 0.0:

            rare_coverage = (
                matched_idf.get(
                    doc_id,
                    0.0
                )
                * inverse_total_query_idf
            )

            prefix_rare_coverage = (
                prefix_matched_idf.get(
                    doc_id,
                    0.0
                )
                * inverse_total_query_idf
            )

        else:

            rare_coverage = 0.0
            prefix_rare_coverage = 0.0

        prefix_coverage = (
            prefix_matched_terms.get(
                doc_id,
                0
            )
            * inverse_unique_terms
        )

        prefix_coordination = (
            prefix_rare_coverage
            * prefix_coverage
        )

        normalized_prefix_bm25 = (
            prefix_bm25_scores.get(
                doc_id,
                0.0
            )
            * inverse_max_prefix_bm25
        )

        prefix_bm25_bonus = (
            normalized_prefix_bm25
            * prefix_coverage
        )

        final_score = (
            normalized_bm25
            + VSM_WEIGHT
            * vsm_value
            + COVERAGE_WEIGHT
            * coverage
            + RARE_WEIGHT
            * rare_coverage
            + PREFIX_WEIGHT
            * prefix_rare_coverage
            + PREFIX_COORD_WEIGHT
            * prefix_coordination
            + PREFIX_BM25_WEIGHT
            * prefix_bm25_bonus
        )

        results.append(
            (
                doc_id,
                final_score
            )
        )

    results.sort(
        key=lambda item: (
            -item[1],
            item[0]
        )
    )

    return results[:k]