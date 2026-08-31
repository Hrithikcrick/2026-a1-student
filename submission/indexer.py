import bz2
import os
import pickle
import re
from array import array
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Tuple

from nltk.stem import PorterStemmer


_INDEX_FILENAME = "inverted_index.pkl.bz2"

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


@lru_cache(maxsize=300000)
def _expand_index_token(
    token: str,
) -> Tuple[str, ...]:

    if token in _STOPWORDS:
        return ()

    stemmed = _stem(token)

    aliases = _ALIAS_TERMS.get(token)

    if not aliases:
        return (stemmed,)

    expanded = [stemmed]

    for alias in aliases:
        if alias in _STOPWORDS:
            continue

        expanded.append(
            _stem(alias)
        )

    return tuple(expanded)


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

    raw_tokens = _TOKEN_RE.findall(
        text
    )

    full_terms = []
    prefix_terms = []

    full_extend = full_terms.extend
    prefix_extend = prefix_terms.extend

    expand = _expand_index_token
    prefix_limit = PREFIX_BASE_TOKENS

    content_position = 0

    for token in raw_tokens:

        expanded = expand(token)

        if not expanded:
            continue

        full_extend(
            expanded
        )

        if content_position < prefix_limit:
            prefix_extend(
                expanded
            )

        content_position += 1

    return (
        Counter(full_terms),
        Counter(prefix_terms),
        len(full_terms),
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



def _pack_integer_postings(
    postings,
):

    packed = {}

    packed_set = packed.__setitem__

    for term, posting in postings.items():

        buffer = bytearray()
        append = buffer.append

        previous_doc = -1

        posting_length = len(posting)

        position = 0

        while position < posting_length:

            integer_doc = posting[position]
            tf = posting[position + 1]

            gap = integer_doc - previous_doc

            if gap < 128:
                append(gap)
            else:
                _write_varint(
                    buffer,
                    gap,
                )

            if tf < 128:
                append(tf)
            else:
                _write_varint(
                    buffer,
                    tf,
                )

            previous_doc = integer_doc

            position += 2

        packed_set(
            term,
            bytes(buffer),
        )

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
        doc_ids = []

        total_doc_len = 0

        postings_get = postings.get
        prefix_postings_get = (
            prefix_postings.get
        )

        doc_ids_append = doc_ids.append

        for integer_doc, (
            doc_id,
            text,
        ) in enumerate(corpus):

            doc_ids_append(
                doc_id
            )

            (
                term_counts,
                prefix_counts,
                length,
            ) = _analyse_document(
                text
            )

            doc_len[doc_id] = length

            total_doc_len += length

            for term, tf in term_counts.items():

                posting = postings_get(
                    term
                )

                if posting is None:

                    postings[term] = [
                        integer_doc,
                        tf,
                    ]

                else:

                    posting.append(
                        integer_doc
                    )

                    posting.append(
                        tf
                    )

            for term, tf in prefix_counts.items():

                posting = (
                    prefix_postings_get(
                        term
                    )
                )

                if posting is None:

                    prefix_postings[term] = [
                        integer_doc,
                        tf,
                    ]

                else:

                    posting.append(
                        integer_doc
                    )

                    posting.append(
                        tf
                    )

        self.postings = postings

        self.prefix_postings = (
            prefix_postings
        )

        self.doc_len = doc_len

        self.doc_ids = doc_ids

        self._integer_postings = True

        self.doc_text = {}

        self.N = len(corpus)

        if self.N:

            self.avg_doc_len = (
                total_doc_len
                / self.N
            )

        else:

            self.avg_doc_len = 0.0


    def document_frequency(
        self,
        term: str,
    ) -> int:

        posting = self.postings.get(
            term
        )

        if posting is None:
            return 0

        if getattr(
            self,
            "_integer_postings",
            False,
        ):

            return len(posting) // 2

        return len(posting)


    def save(
        self,
        index_dir: str,
    ) -> None:

        os.makedirs(
            index_dir,
            exist_ok=True,
        )

        possible_doc_ids = getattr(
            self,
            "doc_ids",
            None,
        )

        if (
            possible_doc_ids is not None
            and len(possible_doc_ids) == self.N
        ):

            doc_ids = list(
                possible_doc_ids
            )

        else:

            doc_ids = list(
                self.doc_len.keys()
            )

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

        if getattr(
            self,
            "_integer_postings",
            False,
        ):

            packed_postings = (
                _pack_integer_postings(
                    self.postings
                )
            )

            packed_prefix_postings = (
                _pack_integer_postings(
                    self.prefix_postings
                )
            )

        else:

            doc_to_int = {
                doc_id: integer_id
                for integer_id, doc_id
                in enumerate(doc_ids)
            }

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

        with bz2.open(
            path,
            "wb",
            compresslevel=5,
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

        with bz2.open(
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