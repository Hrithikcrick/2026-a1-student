import heapq
import math
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Tuple

from submission.indexer import InvertedIndex, tokenize


_INDEX: Optional[InvertedIndex] = None

_DOC_IDS: List[str] = []
_DOC_TO_INT: Dict[str, int] = {}

_POSTINGS_INT: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

_BM25_IDF: Dict[str, float] = {}
_VSM_IDF: Dict[str, float] = {}
_PREFIX_IDF: Dict[str, float] = {}

_DOC_NORMS: np.ndarray = np.empty(0, dtype=np.float64)
_BM25_LENGTH_NORM: np.ndarray = np.empty(0, dtype=np.float64)
_DOC_LEX_RANK: np.ndarray = np.empty(
    0,
    dtype=np.int32,
)

_BM25_ACC: np.ndarray = np.empty(0, dtype=np.float64)
_DOT_ACC: np.ndarray = np.empty(0, dtype=np.float64)
_TOUCH_STAMP: np.ndarray = np.empty(0, dtype=np.int64)
_DOT_STAMP: np.ndarray = np.empty(0, dtype=np.int64)
_QUERY_GENERATION = 0


K1 = 2.4
B = 0.55

CANDIDATE_K = 150

VSM_WEIGHT = 0.225
COVERAGE_WEIGHT = 0.18
RARE_WEIGHT = 0.05
BODY_COORD_WEIGHT = 0.05

PREFIX_WEIGHT = 0.13
PREFIX_COORD_WEIGHT = 0.14

PREFIX_BM25_WEIGHT = 0.16
PREFIX_K1 = 1.2


def build(index: InvertedIndex) -> None:
    global _INDEX
    global _DOC_IDS
    global _DOC_TO_INT
    global _POSTINGS_INT
    global _BM25_IDF
    global _VSM_IDF
    global _PREFIX_IDF
    global _DOC_NORMS
    global _BM25_LENGTH_NORM
    global _DOC_LEX_RANK
    global _BM25_ACC
    global _DOT_ACC
    global _TOUCH_STAMP
    global _DOT_STAMP
    global _QUERY_GENERATION

    _INDEX = index

    if index.N <= 0:
        _DOC_IDS = []
        _DOC_TO_INT = {}
        _POSTINGS_INT = {}
        _BM25_IDF = {}
        _VSM_IDF = {}
        _PREFIX_IDF = {}
        _DOC_NORMS = []
        _BM25_LENGTH_NORM = []
        _DOC_LEX_RANK = np.empty(
            0,
            dtype=np.int32,
        )
        _BM25_ACC = []
        _DOT_ACC = []
        _TOUCH_STAMP = []
        _DOT_STAMP = []
        _QUERY_GENERATION = 0
        return

    possible_doc_ids = getattr(
        index,
        "doc_ids",
        None,
    )

    if (
        possible_doc_ids is not None
        and len(possible_doc_ids) == index.N
    ):
        _DOC_IDS = list(possible_doc_ids)
    else:
        _DOC_IDS = list(index.doc_len.keys())

    _DOC_TO_INT = {
        doc_id: i
        for i, doc_id in enumerate(_DOC_IDS)
    }

    num_docs = len(_DOC_IDS)

    lex_order = sorted(
        range(num_docs),
        key=_DOC_IDS.__getitem__,
    )

    _DOC_LEX_RANK = np.empty(
        num_docs,
        dtype=np.int32,
    )

    for rank, doc_int in enumerate(
        lex_order
    ):
        _DOC_LEX_RANK[doc_int] = rank

    _POSTINGS_INT = {}

    _BM25_IDF = {}
    _VSM_IDF = {}
    _PREFIX_IDF = {}

    _DOC_NORMS = np.zeros(
        num_docs,
        dtype=np.float64,
    )

    _BM25_LENGTH_NORM = np.zeros(
        num_docs,
        dtype=np.float64,
    )

    _BM25_ACC = np.zeros(
        num_docs,
        dtype=np.float64,
    )

    _DOT_ACC = np.zeros(
        num_docs,
        dtype=np.float64,
    )

    _TOUCH_STAMP = np.zeros(
        num_docs,
        dtype=np.int64,
    )

    _DOT_STAMP = np.zeros(
        num_docs,
        dtype=np.int64,
    )

    _QUERY_GENERATION = 0

    N = index.N
    avg_doc_len = index.avg_doc_len

    if avg_doc_len > 0.0:
        for doc_id, doc_length in index.doc_len.items():
            doc_int = _DOC_TO_INT.get(doc_id)

            if doc_int is None:
                continue

            _BM25_LENGTH_NORM[doc_int] = (
                K1
                * (
                    1.0
                    - B
                    + B * (doc_length / avg_doc_len)
                )
            )

    squared_norms = np.zeros(
        num_docs,
        dtype=np.float64,
    )

    for term, posting in index.postings.items():
        df = len(posting)

        if df == 0:
            continue

        bm25_idf = math.log(
            ((N - df + 0.5) / (df + 0.5)) + 1.0
        )

        vsm_idf = math.log(
            N / df
        )

        _BM25_IDF[term] = bm25_idf
        _VSM_IDF[term] = vsm_idf

        posting_items = [
            (
                _DOC_TO_INT[doc_id],
                tf,
            )
            for doc_id, tf in posting.items()
            if doc_id in _DOC_TO_INT
        ]

        if not posting_items:
            continue

        docs = np.fromiter(
            (
                item[0]
                for item in posting_items
            ),
            dtype=np.int32,
            count=len(posting_items),
        )

        tfs = np.fromiter(
            (
                item[1]
                for item in posting_items
            ),
            dtype=np.float64,
            count=len(posting_items),
        )

        _POSTINGS_INT[term] = (
            docs,
            tfs,
        )

        if vsm_idf != 0.0:
            weights = (
                tfs
                * vsm_idf
            )

            squared_norms[docs] += (
                weights
                * weights
            )

    np.sqrt(
        squared_norms,
        out=_DOC_NORMS,
    )

    prefix_postings = getattr(
        index,
        "prefix_postings",
        {},
    )

    for term, posting in prefix_postings.items():
        df = len(posting)

        if df == 0:
            continue

        _PREFIX_IDF[term] = math.log(
            ((N - df + 0.5) / (df + 0.5)) + 1.0
        )


def _exact_topk(
    docs: np.ndarray,
    scores: np.ndarray,
    k: int,
    lex_rank: np.ndarray,
) -> Dict[int, float]:

    n = docs.size

    if n == 0 or k <= 0:
        return {}

    if n <= k:
        selected_docs = docs
        selected_scores = scores

    else:
        partition = np.argpartition(
            scores,
            -k,
        )[-k:]

        threshold = np.min(
            scores[partition]
        )

        mask = (
            scores
            >= threshold
        )

        selected_docs = docs[
            mask
        ]

        selected_scores = scores[
            mask
        ]

    selected_lex = lex_rank[
        selected_docs
    ]

    order = np.lexsort(
        (
            selected_lex,
            -selected_scores,
        )
    )

    if order.size > k:
        order = order[:k]

    final_docs = selected_docs[
        order
    ]

    final_scores = selected_scores[
        order
    ]

    return {
        int(doc_int): float(score)
        for doc_int, score
        in zip(
            final_docs,
            final_scores,
        )
    }


def score(
    query: str,
    k: int,
) -> List[Tuple[str, float]]:

    global _QUERY_GENERATION

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

    postings_int = _POSTINGS_INT

    original_postings = index.postings

    prefix_postings = getattr(
        index,
        "prefix_postings",
        {},
    )

    bm25_idfs = _BM25_IDF
    vsm_idfs = _VSM_IDF
    prefix_idfs = _PREFIX_IDF

    doc_norms = _DOC_NORMS
    length_norms = _BM25_LENGTH_NORM

    doc_ids = _DOC_IDS
    doc_to_int = _DOC_TO_INT
    doc_lex_rank = _DOC_LEX_RANK

    bm25_acc = _BM25_ACC
    dot_acc = _DOT_ACC
    touch_stamp = _TOUCH_STAMP
    dot_stamp = _DOT_STAMP

    _QUERY_GENERATION += 1
    generation = _QUERY_GENERATION

    active_terms = []

    query_norm_squared = 0.0
    total_query_idf = 0.0

    k1_plus_one = K1 + 1.0

    for term, qtf in query_tf.items():
        posting = postings_int.get(term)

        if posting is None:
            continue

        docs, tfs = posting

        bm25_idf = bm25_idfs.get(
            term,
            0.0,
        )

        vsm_idf = vsm_idfs.get(
            term,
            0.0,
        )

        active_terms.append(
            (
                term,
                bm25_idf,
            )
        )

        total_query_idf += bm25_idf

        query_weight = (
            qtf * vsm_idf
        )

        query_norm_squared += (
            query_weight
            * query_weight
        )

        bm25_multiplier = (
            qtf
            * bm25_idf
            * k1_plus_one
        )

        new_mask = (
            touch_stamp[docs]
            != generation
        )

        if np.any(new_mask):
            new_docs = docs[
                new_mask
            ]

            touch_stamp[new_docs] = (
                generation
            )

            bm25_acc[new_docs] = 0.0
            dot_acc[new_docs] = 0.0

        bm25_acc[docs] += (
            bm25_multiplier
            * tfs
            / (
                tfs
                + length_norms[docs]
            )
        )

        if vsm_idf != 0.0:
            vsm_multiplier = (
                query_weight
                * vsm_idf
            )

            dot_acc[docs] += (
                vsm_multiplier
                * tfs
            )

            dot_stamp[docs] = (
                generation
            )

    touched_array = np.flatnonzero(
        touch_stamp == generation
    )

    if touched_array.size == 0:
        return []

    bm25_values = bm25_acc[
        touched_array
    ]

    bm25_top_scores = _exact_topk(
        touched_array,
        bm25_values,
        CANDIDATE_K,
        doc_lex_rank,
    )

    query_norm = math.sqrt(
        query_norm_squared
    )

    if query_norm > 0.0:
        inverse_query_norm = (
            1.0 / query_norm
        )

        valid_vsm_mask = (
            (
                dot_stamp[touched_array]
                == generation
            )
            & (
                doc_norms[touched_array]
                > 0.0
            )
        )

        vsm_docs = touched_array[
            valid_vsm_mask
        ]

        vsm_values = (
            dot_acc[vsm_docs]
            * inverse_query_norm
            / doc_norms[vsm_docs]
        )

        vsm_top_scores = _exact_topk(
            vsm_docs,
            vsm_values,
            CANDIDATE_K,
            doc_lex_rank,
        )

    else:
        vsm_top_scores = {}

    candidates = set(
        bm25_top_scores
    )

    candidates.update(
        vsm_top_scores
    )

    candidate_count = len(
        candidates
    )

    matched_terms: Dict[int, int] = {}
    matched_idf: Dict[int, float] = {}

    matched_terms_get = matched_terms.get
    matched_idf_get = matched_idf.get

    for term, bm25_idf in active_terms:
        posting = original_postings[term]

        if len(posting) < candidate_count:

            for doc_id in posting:
                doc_int = doc_to_int.get(
                    doc_id
                )

                if (
                    doc_int is None
                    or doc_int not in candidates
                ):
                    continue

                matched_terms[doc_int] = (
                    matched_terms_get(
                        doc_int,
                        0,
                    )
                    + 1
                )

                matched_idf[doc_int] = (
                    matched_idf_get(
                        doc_int,
                        0.0,
                    )
                    + bm25_idf
                )

        else:
            for doc_int in candidates:

                if doc_ids[doc_int] not in posting:
                    continue

                matched_terms[doc_int] = (
                    matched_terms_get(
                        doc_int,
                        0,
                    )
                    + 1
                )

                matched_idf[doc_int] = (
                    matched_idf_get(
                        doc_int,
                        0.0,
                    )
                    + bm25_idf
                )

    max_bm25 = max(
        bm25_top_scores.values(),
        default=1.0,
    )

    if max_bm25 <= 0.0:
        max_bm25 = 1.0

    inverse_max_bm25 = (
        1.0 / max_bm25
    )

    unique_query_terms = len(
        query_tf
    )

    inverse_unique_terms = (
        1.0 / unique_query_terms
    )

    if total_query_idf > 0.0:
        inverse_total_query_idf = (
            1.0 / total_query_idf
        )
    else:
        inverse_total_query_idf = 0.0

    prefix_matched_terms: Dict[int, int] = {}
    prefix_matched_idf: Dict[int, float] = {}
    prefix_bm25_scores: Dict[int, float] = {}

    prefix_matched_terms_get = (
        prefix_matched_terms.get
    )

    prefix_matched_idf_get = (
        prefix_matched_idf.get
    )

    prefix_bm25_scores_get = (
        prefix_bm25_scores.get
    )

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

        prefix_idf = prefix_idfs.get(
            term,
            0.0,
        )

        full_idf = bm25_idfs.get(
            term,
            0.0,
        )

        if prefix_idf <= 0.0:
            continue

        if len(prefix_posting) < candidate_count:

            for doc_id, tf in prefix_posting.items():

                doc_int = doc_to_int.get(
                    doc_id
                )

                if (
                    doc_int is None
                    or doc_int not in candidates
                ):
                    continue

                prefix_matched_terms[doc_int] = (
                    prefix_matched_terms_get(
                        doc_int,
                        0,
                    )
                    + 1
                )

                prefix_matched_idf[doc_int] = (
                    prefix_matched_idf_get(
                        doc_int,
                        0.0,
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

                prefix_bm25_scores[doc_int] = (
                    prefix_bm25_scores_get(
                        doc_int,
                        0.0,
                    )
                    + qtf
                    * prefix_idf
                    * tf_component
                )

        else:
            for doc_int in candidates:

                doc_id = doc_ids[doc_int]

                tf = prefix_posting.get(
                    doc_id
                )

                if tf is None:
                    continue

                prefix_matched_terms[doc_int] = (
                    prefix_matched_terms_get(
                        doc_int,
                        0,
                    )
                    + 1
                )

                prefix_matched_idf[doc_int] = (
                    prefix_matched_idf_get(
                        doc_int,
                        0.0,
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

                prefix_bm25_scores[doc_int] = (
                    prefix_bm25_scores_get(
                        doc_int,
                        0.0,
                    )
                    + qtf
                    * prefix_idf
                    * tf_component
                )

    max_prefix_bm25 = max(
        prefix_bm25_scores.values(),
        default=1.0,
    )

    if max_prefix_bm25 <= 0.0:
        max_prefix_bm25 = 1.0

    inverse_max_prefix_bm25 = (
        1.0 / max_prefix_bm25
    )

    results: List[Tuple[str, float]] = []

    bm25_top_scores_get = (
        bm25_top_scores.get
    )

    vsm_top_scores_get = (
        vsm_top_scores.get
    )

    for doc_int in candidates:

        normalized_bm25 = (
            bm25_top_scores_get(
                doc_int,
                0.0,
            )
            * inverse_max_bm25
        )

        vsm_value = (
            vsm_top_scores_get(
                doc_int,
                0.0,
            )
        )

        coverage = (
            matched_terms_get(
                doc_int,
                0,
            )
            * inverse_unique_terms
        )

        if inverse_total_query_idf > 0.0:

            rare_coverage = (
                matched_idf_get(
                    doc_int,
                    0.0,
                )
                * inverse_total_query_idf
            )

            prefix_rare_coverage = (
                prefix_matched_idf_get(
                    doc_int,
                    0.0,
                )
                * inverse_total_query_idf
            )

        else:
            rare_coverage = 0.0
            prefix_rare_coverage = 0.0

        body_coordination = (
            coverage
            * rare_coverage
        )

        prefix_coverage = (
            prefix_matched_terms_get(
                doc_int,
                0,
            )
            * inverse_unique_terms
        )

        prefix_coordination = (
            prefix_rare_coverage
            * prefix_coverage
        )

        normalized_prefix_bm25 = (
            prefix_bm25_scores_get(
                doc_int,
                0.0,
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
            + BODY_COORD_WEIGHT
            * body_coordination
            + PREFIX_WEIGHT
            * prefix_rare_coverage
            + PREFIX_COORD_WEIGHT
            * prefix_coordination
            + PREFIX_BM25_WEIGHT
            * prefix_bm25_bonus
        )

        results.append(
            (
                doc_ids[doc_int],
                final_score,
            )
        )

    results.sort(
        key=lambda item: (
            -item[1],
            item[0],
        )
    )

    return results[:k]