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

PREFIX_BASE_TOKENS = 16


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "for",
    "from",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "may",
    "might",
    "more",
    "most",
    "my",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "out",
    "over",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "under",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
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


_ALIAS_TERMS = {
    "sarscov2": [
        "coronavirus",
        "sars",
        "cov",
    ],
    "covid19": [
        "covid",
        "coronavirus",
    ],
    "2019ncov": [
        "ncov",
        "coronavirus",
    ],
    "coronavirus": [
        "covid19",
    ],
    "mrna": [
        "messenger",
        "rna",
    ],
    "tcell": [
        "cell",
    ],
    "bcell": [
        "cell",
    ],
}


@lru_cache(maxsize=200000)
def _stem(token: str) -> str:
    return _STEMMER.stem(token)


def _prepare_text(text: str) -> str:
    text = text.lower()

    for source, target in _COMPOUND_REPLACEMENTS.items():
        text = text.replace(
            source,
            target,
        )

    return text


def _expanded_token(token: str) -> List[str]:
    result = [
        _stem(token)
    ]

    aliases = _ALIAS_TERMS.get(token)

    if aliases:
        for alias in aliases:
            if alias not in _STOPWORDS:
                result.append(
                    _stem(alias)
                )

    return result


def tokenize(text: str) -> List[str]:
    text = _prepare_text(text)

    raw_tokens = _TOKEN_RE.findall(text)

    output = []

    for token in raw_tokens:
        if token in _STOPWORDS:
            continue

        output.extend(
            _expanded_token(token)
        )

    return output


def _tokenize_with_prefix(
    text: str
) -> Tuple[List[str], List[str]]:

    text = _prepare_text(text)

    raw_tokens = _TOKEN_RE.findall(text)

    output = []
    prefix = []

    content_position = 0

    for token in raw_tokens:
        if token in _STOPWORDS:
            continue

        expanded = _expanded_token(token)

        output.extend(expanded)

        if content_position < PREFIX_BASE_TOKENS:
            prefix.extend(expanded)

        content_position += 1

    return output, prefix


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

        postings: Dict[
            str,
            Dict[str, int]
        ] = {}

        prefix_postings: Dict[
            str,
            Dict[str, int]
        ] = {}

        doc_len: Dict[
            str,
            int
        ] = {}

        total_doc_len = 0

        for doc_id, text in corpus:

            tokens, prefix_tokens = (
                _tokenize_with_prefix(text)
            )

            length = len(tokens)

            doc_len[doc_id] = length

            total_doc_len += length

            term_counts = Counter(tokens)

            prefix_counts = Counter(
                prefix_tokens
            )

            for term, tf in term_counts.items():

                posting = postings.get(term)

                if posting is None:
                    postings[term] = {
                        doc_id: tf
                    }
                else:
                    posting[doc_id] = tf

            for term, tf in prefix_counts.items():

                posting = prefix_postings.get(
                    term
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

        if self.N > 0:
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

        return len(
            self.postings.get(
                term,
                {},
            )
        )


    def _compact(
        self,
        postings,
        doc_to_int,
    ):

        compact = {}

        for term, posting in postings.items():

            integer_doc_ids = array(
                "I",
                (
                    doc_to_int[doc_id]
                    for doc_id
                    in posting.keys()
                ),
            )

            term_frequencies = array(
                "I",
                (
                    tf
                    for tf
                    in posting.values()
                ),
            )

            compact[term] = (
                integer_doc_ids,
                term_frequencies,
            )

        return compact


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

        doc_lengths = array(
            "I",
            (
                self.doc_len[doc_id]
                for doc_id in doc_ids
            ),
        )

        compact_postings = self._compact(
            self.postings,
            doc_to_int,
        )

        compact_prefix_postings = self._compact(
            self.prefix_postings,
            doc_to_int,
        )

        state = {
            "doc_ids": doc_ids,
            "doc_lengths": doc_lengths,
            "postings": compact_postings,
            "prefix_postings": (
                compact_prefix_postings
            ),
            "N": self.N,
            "avg_doc_len": self.avg_doc_len,
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

        def restore_postings(
            compact_postings
        ):

            restored = {}

            for term, packed in (
                compact_postings.items()
            ):

                integer_doc_ids, term_frequencies = (
                    packed
                )

                posting = {}

                for integer_doc_id, tf in zip(
                    integer_doc_ids,
                    term_frequencies,
                ):

                    doc_id = doc_ids[
                        integer_doc_id
                    ]

                    posting[doc_id] = int(tf)

                restored[term] = posting

            return restored

        index.postings = restore_postings(
            state["postings"]
        )

        index.prefix_postings = (
            restore_postings(
                state.get(
                    "prefix_postings",
                    {},
                )
            )
        )

        index.N = state["N"]

        index.avg_doc_len = state[
            "avg_doc_len"
        ]

        index.doc_text = {}

        return index