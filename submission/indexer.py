import gzip
import os
import pickle
import re
from array import array
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Tuple

from nltk.stem import PorterStemmer


_INDEX_FILENAME = "inverted_index.pkl.gz"

_STEMMER = PorterStemmer()
_TOKEN_RE = re.compile(r"[a-z0-9]+")

PREFIX_BASE_TOKENS = 8


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
    "which", "while", "who", "why", "will", "with", "would",
    "you", "your", "yours",
}


_COMPOUND_REPLACEMENTS = {
    "sars-cov-2": "sarscov2",
    "sars cov 2": "sarscov2",
    "sars-cov2": "sarscov2",
    "sarscov2": "sarscov2",
    "covid-19": "covid19",
    "covid 19": "covid19",
    "covid19": "covid19",
    "2019-ncov": "2019ncov",
    "2019 ncov": "2019ncov",
    "m-rna": "mrna",
    "m rna": "mrna",
    "t-cell": "tcell",
    "t cell": "tcell",
    "b-cell": "bcell",
    "b cell": "bcell",
}


_COMPOUND_RE = re.compile(
    "|".join(
        re.escape(key)
        for key in sorted(
            _COMPOUND_REPLACEMENTS,
            key=len,
            reverse=True,
        )
    )
)


_ALIAS_TERMS = {
    "sarscov2": ["coronavirus", "sars", "cov"],
    "covid19": ["covid", "coronavirus"],
    "2019ncov": ["ncov", "coronavirus"],
    "coronavirus": ["covid19"],
    "mrna": ["messenger", "rna"],
    "tcell": ["cell"],
    "bcell": ["cell"],
}


@lru_cache(maxsize=300000)
def _stem(token: str) -> str:
    return _STEMMER.stem(token)


def _prepare_text(text: str) -> str:
    text = text.lower()

    return _COMPOUND_RE.sub(
        lambda match: _COMPOUND_REPLACEMENTS[
            match.group(0)
        ],
        text,
    )


def tokenize(text: str) -> List[str]:
    text = _prepare_text(text)

    raw_tokens = _TOKEN_RE.findall(text)

    output = []

    for token in raw_tokens:
        if token in _STOPWORDS:
            continue

        output.append(_stem(token))

        aliases = _ALIAS_TERMS.get(token)

        if aliases:
            for alias in aliases:
                if alias not in _STOPWORDS:
                    output.append(
                        _stem(alias)
                    )

    return output


def _analyse_document(
    text: str,
) -> Tuple[Counter, Counter, int]:

    text = _prepare_text(text)

    raw_tokens = _TOKEN_RE.findall(text)

    full_counts = Counter()
    prefix_counts = Counter()

    expanded_length = 0
    content_position = 0

    for token in raw_tokens:
        if token in _STOPWORDS:
            continue

        stemmed = _stem(token)

        full_counts[stemmed] += 1
        expanded_length += 1

        in_prefix = (
            content_position
            < PREFIX_BASE_TOKENS
        )

        if in_prefix:
            prefix_counts[stemmed] += 1

        aliases = _ALIAS_TERMS.get(token)

        if aliases:
            for alias in aliases:
                if alias in _STOPWORDS:
                    continue

                alias_stem = _stem(alias)

                full_counts[alias_stem] += 1
                expanded_length += 1

                if in_prefix:
                    prefix_counts[
                        alias_stem
                    ] += 1

        content_position += 1

    return (
        full_counts,
        prefix_counts,
        expanded_length,
    )


def _write_varint(
    buffer: bytearray,
    value: int,
) -> None:

    while value >= 128:
        buffer.append(
            (value & 127) | 128
        )
        value >>= 7

    buffer.append(value)


def _pack_postings(
    postings,
    doc_to_int,
):

    packed = {}

    for term, posting in postings.items():

        buffer = bytearray()

        previous_doc = -1

        for doc_id, tf in posting.items():

            integer_doc = doc_to_int[
                doc_id
            ]

            gap = (
                integer_doc
                - previous_doc
            )

            _write_varint(
                buffer,
                gap,
            )

            _write_varint(
                buffer,
                int(tf),
            )

            previous_doc = integer_doc

        packed[term] = bytes(buffer)

    return packed


def _unpack_postings(
    packed_postings,
    doc_ids,
):

    result = {}

    for term, data in packed_postings.items():

        posting = {}

        position = 0
        data_length = len(data)

        previous_doc = -1

        while position < data_length:

            gap = 0
            shift = 0

            while True:
                byte = data[position]
                position += 1

                gap |= (
                    byte & 127
                ) << shift

                if byte < 128:
                    break

                shift += 7

            tf = 0
            shift = 0

            while True:
                byte = data[position]
                position += 1

                tf |= (
                    byte & 127
                ) << shift

                if byte < 128:
                    break

                shift += 7

            integer_doc = (
                previous_doc
                + gap
            )

            posting[
                doc_ids[integer_doc]
            ] = tf

            previous_doc = integer_doc

        result[term] = posting

    return result


class InvertedIndex:

    def __init__(self):

        self.postings: Dict[
            str,
            Dict[str, int]
        ] = {}

        self.prefix_postings: Dict[
            str,
            Dict[str, int]
        ] = {}

        self.doc_len: Dict[
            str,
            int
        ] = {}

        self.doc_text: Dict[
            str,
            str
        ] = {}

        self.N: int = 0
        self.avg_doc_len: float = 0.0


    def build(
        self,
        corpus: List[Tuple[str, str]],
    ) -> None:

        postings = {}
        prefix_postings = {}

        doc_len = {}

        total_doc_len = 0

        postings_get = postings.get

        prefix_postings_get = (
            prefix_postings.get
        )

        for doc_id, text in corpus:

            (
                term_counts,
                prefix_counts,
                length,
            ) = _analyse_document(text)

            doc_len[doc_id] = length

            total_doc_len += length

            for term, tf in term_counts.items():

                posting = postings_get(
                    term
                )

                if posting is None:
                    postings[term] = {
                        doc_id: tf
                    }
                else:
                    posting[doc_id] = tf

            for term, tf in prefix_counts.items():

                posting = (
                    prefix_postings_get(
                        term
                    )
                )

                if posting is None:
                    prefix_postings[term] = {
                        doc_id: tf
                    }
                else:
                    posting[doc_id] = tf

        self.postings = postings

        self.prefix_postings = (
            prefix_postings
        )

        self.doc_len = doc_len

        self.doc_text = {}

        self.N = len(corpus)

        if self.N:
            self.avg_doc_len = (
                total_doc_len / self.N
            )
        else:
            self.avg_doc_len = 0.0


    def document_frequency(
        self,
        term: str,
    ) -> int:

        return len(
            self.postings.get(
                term,
                {},
            )
        )


    def save(
        self,
        index_dir: str,
    ) -> None:

        os.makedirs(
            index_dir,
            exist_ok=True,
        )

        doc_ids = list(
            self.doc_len.keys()
        )

        doc_to_int = {
            doc_id: integer_id
            for integer_id, doc_id
            in enumerate(doc_ids)
        }

        max_doc_length = max(
            self.doc_len.values(),
            default=0,
        )

        if max_doc_length <= 65535:
            doc_length_type = "H"
        else:
            doc_length_type = "I"

        doc_lengths = array(
            doc_length_type,
            (
                self.doc_len[doc_id]
                for doc_id in doc_ids
            ),
        )

        packed_postings = (
            _pack_postings(
                self.postings,
                doc_to_int,
            )
        )

        packed_prefix_postings = (
            _pack_postings(
                self.prefix_postings,
                doc_to_int,
            )
        )

        state = {
            "format_version": 2,
            "doc_ids": doc_ids,
            "doc_lengths": doc_lengths,
            "postings": packed_postings,
            "prefix_postings":
                packed_prefix_postings,
            "N": self.N,
            "avg_doc_len":
                self.avg_doc_len,
        }

        path = os.path.join(
            index_dir,
            _INDEX_FILENAME,
        )

        with gzip.open(
            path,
            "wb",
            compresslevel=1,
        ) as f:

            pickle.dump(
                state,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


    @classmethod
    def load(
        cls,
        index_dir: str,
    ) -> "InvertedIndex":

        path = os.path.join(
            index_dir,
            _INDEX_FILENAME,
        )

        with gzip.open(
            path,
            "rb",
        ) as f:

            state = pickle.load(f)

        index = cls()

        doc_ids = state[
            "doc_ids"
        ]

        doc_lengths = state[
            "doc_lengths"
        ]

        index.doc_len = {
            doc_id: int(length)
            for doc_id, length
            in zip(
                doc_ids,
                doc_lengths,
            )
        }

        index.postings = (
            _unpack_postings(
                state["postings"],
                doc_ids,
            )
        )

        index.prefix_postings = (
            _unpack_postings(
                state.get(
                    "prefix_postings",
                    {},
                ),
                doc_ids,
            )
        )

        index.N = state[
            "N"
        ]

        index.avg_doc_len = state[
            "avg_doc_len"
        ]

        index.doc_text = {}

        return index