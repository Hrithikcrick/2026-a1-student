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

        previous_doc = -1

        posting_length = len(posting)

        position = 0

        while position < posting_length:

            integer_doc = posting[position]
            tf = posting[position + 1]

            gap = (
                integer_doc
                - previous_doc
            )

            has_tf = (
                0 if tf == 1 else 1
            )

            encoded_gap = (
                (gap << 1)
                | has_tf
            )

            _write_varint(
                buffer,
                encoded_gap,
            )

            if has_tf:

                _write_varint(
                    buffer,
                    tf - 1,
                )

            previous_doc = integer_doc

            position += 2

        packed_set(
            term,
            bytes(buffer),
        )

    return packed


def _unpack_postings_compact(
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

            encoded_gap = 0
            shift = 0

            while True:

                byte = data[position]
                position += 1

                encoded_gap |= (
                    byte & 127
                ) << shift

                if byte < 128:
                    break

                shift += 7

            has_tf = (
                encoded_gap & 1
            )

            gap = (
                encoded_gap >> 1
            )

            if has_tf:

                stored_tf = 0
                shift = 0

                while True:

                    byte = data[position]
                    position += 1

                    stored_tf |= (
                        byte & 127
                    ) << shift

                    if byte < 128:
                        break

                    shift += 7

                tf = stored_tf + 1

            else:

                tf = 1

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

        posting_terms = list(
            packed_postings.keys()
        )

        body_lengths = array(
            "I",
            (
                len(
                    packed_postings[term]
                )
                for term in posting_terms
            ),
        )

        body_blob = b"".join(
            packed_postings[term]
            for term in posting_terms
        )

        term_to_int = {
            term: integer_term
            for integer_term, term
            in enumerate(posting_terms)
        }

        prefix_items = []

        for term, data in (
            packed_prefix_postings.items()
        ):
            integer_term = (
                term_to_int.get(term)
            )

            if integer_term is None:
                raise ValueError(
                    "Prefix term missing from body vocabulary"
                )

            prefix_items.append(
                (
                    integer_term,
                    data,
                )
            )

        prefix_items.sort(
            key=lambda item: item[0]
        )

        prefix_bitmap = bytearray(
            (
                len(posting_terms)
                + 7
            )
            // 8
        )

        prefix_lengths = array(
            "I"
        )

        prefix_parts = []

        for integer_term, data in (
            prefix_items
        ):
            prefix_bitmap[
                integer_term >> 3
            ] |= (
                1
                << (
                    integer_term
                    & 7
                )
            )

            prefix_lengths.append(
                len(data)
            )

            prefix_parts.append(
                data
            )

        prefix_blob = b"".join(
            prefix_parts
        )

        state = {
            "format_version": 4,
            "doc_ids": doc_ids,
            "doc_lengths": doc_lengths,
            "posting_terms":
                posting_terms,
            "body_lengths":
                body_lengths,
            "body_blob":
                body_blob,
            "prefix_bitmap":
                bytes(prefix_bitmap),
            "prefix_lengths":
                prefix_lengths,
            "prefix_blob":
                prefix_blob,
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
            compresslevel=9,
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

        format_version = state.get(
            "format_version",
            2,
        )

        if (
            format_version >= 4
            and "body_blob" in state
        ):
            posting_terms = state[
                "posting_terms"
            ]

            body_lengths = state[
                "body_lengths"
            ]

            body_blob = state[
                "body_blob"
            ]

            packed_postings = {}

            position = 0

            for term, length in zip(
                posting_terms,
                body_lengths,
            ):
                next_position = (
                    position
                    + int(length)
                )

                packed_postings[
                    term
                ] = body_blob[
                    position:
                    next_position
                ]

                position = (
                    next_position
                )

            if position != len(
                body_blob
            ):
                raise ValueError(
                    "Invalid body posting blob"
                )

            prefix_bitmap = state[
                "prefix_bitmap"
            ]

            prefix_lengths = state[
                "prefix_lengths"
            ]

            prefix_blob = state[
                "prefix_blob"
            ]

            packed_prefix_postings = {}

            prefix_position = 0
            prefix_length_position = 0

            for integer_term, term in (
                enumerate(
                    posting_terms
                )
            ):
                if not (
                    prefix_bitmap[
                        integer_term >> 3
                    ]
                    & (
                        1
                        << (
                            integer_term
                            & 7
                        )
                    )
                ):
                    continue

                length = int(
                    prefix_lengths[
                        prefix_length_position
                    ]
                )

                next_position = (
                    prefix_position
                    + length
                )

                packed_prefix_postings[
                    term
                ] = prefix_blob[
                    prefix_position:
                    next_position
                ]

                prefix_position = (
                    next_position
                )

                prefix_length_position += 1

            if (
                prefix_position
                != len(prefix_blob)
            ):
                raise ValueError(
                    "Invalid prefix posting blob"
                )

            if (
                prefix_length_position
                != len(prefix_lengths)
            ):
                raise ValueError(
                    "Invalid prefix posting lengths"
                )

        else:
            packed_postings = state[
                "postings"
            ]

            packed_prefix_postings = (
                state.get(
                    "prefix_postings",
                    {},
                )
            )

        if format_version >= 3:

            unpacker = (
                _unpack_postings_compact
            )

        else:

            unpacker = (
                _unpack_postings
            )

        index.postings = unpacker(
            packed_postings,
            doc_ids,
        )

        index.prefix_postings = unpacker(
            packed_prefix_postings,
            doc_ids,
        )

        index.N = state[
            "N"
        ]

        index.avg_doc_len = state[
            "avg_doc_len"
        ]

        index.doc_text = {}

        return index
_HYBRID_V7_ACTIVE = True

_V7_LEGACY_LOAD = InvertedIndex.load


class _V7BitWriter:

    def __init__(self):
        self.data = bytearray()
        self.buffer = 0
        self.bits = 0

    def write_bits(
        self,
        value,
        width,
    ):
        if width <= 0:
            return

        self.buffer = (
            self.buffer << width
        ) | value

        self.bits += width

        while self.bits >= 8:
            shift = self.bits - 8

            self.data.append(
                (
                    self.buffer
                    >> shift
                )
                & 255
            )

            self.bits -= 8

            if self.bits:
                self.buffer &= (
                    (1 << self.bits)
                    - 1
                )
            else:
                self.buffer = 0

    def write_bounded(
        self,
        value,
        count,
    ):
        if count <= 1:
            return

        width = (
            count - 1
        ).bit_length()

        cutoff = (
            (1 << width)
            - count
        )

        if value < cutoff:
            self.write_bits(
                value,
                width - 1,
            )
        else:
            self.write_bits(
                value + cutoff,
                width,
            )

    def write_gamma(
        self,
        value,
    ):
        if value <= 0:
            raise ValueError(value)

        width = value.bit_length()

        self.write_bits(
            value,
            2 * width - 1,
        )

    def finish(self):
        if self.bits:
            self.data.append(
                (
                    self.buffer
                    << (
                        8 - self.bits
                    )
                )
                & 255
            )

            self.buffer = 0
            self.bits = 0

        return bytes(
            self.data
        )


class _V7BitReader:

    def __init__(
        self,
        data,
    ):
        self.data = data
        self.position = 0
        self.buffer = 0
        self.bits = 0

    def read_bits(
        self,
        width,
    ):
        if width <= 0:
            return 0

        while self.bits < width:
            if self.position >= len(
                self.data
            ):
                raise ValueError(
                    "Unexpected end of bit stream"
                )

            self.buffer = (
                self.buffer << 8
            ) | self.data[
                self.position
            ]

            self.position += 1
            self.bits += 8

        shift = self.bits - width

        value = (
            self.buffer
            >> shift
        ) & (
            (1 << width)
            - 1
        )

        self.bits -= width

        if self.bits:
            self.buffer &= (
                (1 << self.bits)
                - 1
            )
        else:
            self.buffer = 0

        return value

    def read_bounded(
        self,
        count,
    ):
        if count <= 1:
            return 0

        width = (
            count - 1
        ).bit_length()

        cutoff = (
            (1 << width)
            - count
        )

        value = self.read_bits(
            width - 1
        )

        if value < cutoff:
            return value

        return (
            (
                value << 1
            )
            | self.read_bits(1)
        ) - cutoff

    def read_gamma(self):
        zeros = 0

        while self.read_bits(1) == 0:
            zeros += 1

        value = 1

        if zeros:
            value = (
                value << zeros
            ) | self.read_bits(
                zeros
            )

        return value


def _v7_encode_varints(
    values,
):
    buffer = bytearray()

    for value in values:
        _write_varint(
            buffer,
            int(value),
        )

    return bytes(buffer)


def _v7_decode_varints(
    data,
    count,
):
    result = []
    position = 0

    for _ in range(count):
        value = 0
        shift = 0

        while True:
            if position >= len(data):
                raise ValueError(
                    "Invalid varint stream"
                )

            byte = data[
                position
            ]

            position += 1

            value |= (
                byte & 127
            ) << shift

            if byte < 128:
                break

            shift += 7

        result.append(
            value
        )

    if position != len(data):
        raise ValueError(
            "Invalid varint metadata"
        )

    return result


def _v7_gap_encode(
    values,
):
    buffer = bytearray()
    previous = -1

    for value in values:
        _write_varint(
            buffer,
            value - previous,
        )

        previous = value

    return bytes(buffer)


def _v7_gap_decode(
    data,
    count,
):
    result = []
    position = 0
    previous = -1

    for _ in range(count):
        gap = 0
        shift = 0

        while True:
            if position >= len(data):
                raise ValueError(
                    "Invalid gap stream"
                )

            byte = data[
                position
            ]

            position += 1

            gap |= (
                byte & 127
            ) << shift

            if byte < 128:
                break

            shift += 7

        previous += gap

        result.append(
            previous
        )

    if position != len(data):
        raise ValueError(
            "Invalid gap payload"
        )

    return result


def _v7_bitmap_encode(
    values,
    universe,
):
    output = bytearray(
        (
            universe + 7
        )
        // 8
    )

    for value in values:
        output[
            value >> 3
        ] |= (
            1
            << (
                value & 7
            )
        )

    return bytes(output)


def _v7_bitmap_decode(
    data,
    universe,
    count,
):
    output = []

    for value in range(
        universe
    ):
        if data[
            value >> 3
        ] & (
            1
            << (
                value & 7
            )
        ):
            output.append(
                value
            )

    if len(output) != count:
        raise ValueError(
            "Invalid bitmap payload"
        )

    return output


def _v7_interpolative_encode(
    values,
    universe,
):
    if not values:
        return b""

    writer = _V7BitWriter()

    stack = [
        (
            0,
            len(values) - 1,
            0,
            universe - 1,
        )
    ]

    while stack:
        (
            left,
            right,
            low,
            high,
        ) = stack.pop()

        if left > right:
            continue

        middle = (
            left + right
        ) // 2

        minimum = (
            low
            + middle
            - left
        )

        maximum = (
            high
            - (
                right - middle
            )
        )

        count = (
            maximum
            - minimum
            + 1
        )

        middle_value = values[
            middle
        ]

        writer.write_bounded(
            middle_value
            - minimum,
            count,
        )

        stack.append(
            (
                middle + 1,
                right,
                middle_value + 1,
                high,
            )
        )

        stack.append(
            (
                left,
                middle - 1,
                low,
                middle_value - 1,
            )
        )

    return writer.finish()


def _v7_interpolative_decode(
    data,
    count,
    universe,
):
    if count == 0:
        return []

    reader = _V7BitReader(
        data
    )

    values = [
        0
    ] * count

    stack = [
        (
            0,
            count - 1,
            0,
            universe - 1,
        )
    ]

    while stack:
        (
            left,
            right,
            low,
            high,
        ) = stack.pop()

        if left > right:
            continue

        middle = (
            left + right
        ) // 2

        minimum = (
            low
            + middle
            - left
        )

        maximum = (
            high
            - (
                right - middle
            )
        )

        possible = (
            maximum
            - minimum
            + 1
        )

        middle_value = (
            minimum
            + reader.read_bounded(
                possible
            )
        )

        values[
            middle
        ] = middle_value

        stack.append(
            (
                middle + 1,
                right,
                middle_value + 1,
                high,
            )
        )

        stack.append(
            (
                left,
                middle - 1,
                low,
                middle_value - 1,
            )
        )

    return values


def _v7_choose_codec(
    values,
    universe,
):
    gap_blob = _v7_gap_encode(
        values
    )

    interp_blob = (
        _v7_interpolative_encode(
            values,
            universe,
        )
    )

    code = 0
    blob = gap_blob

    if len(interp_blob) < len(
        blob
    ):
        code = 1
        blob = interp_blob

    bitmap_size = (
        universe + 7
    ) // 8

    if bitmap_size < len(blob):
        code = 2
        blob = _v7_bitmap_encode(
            values,
            universe,
        )

    return code, blob


def _v7_decode_codec(
    code,
    data,
    count,
    universe,
):
    if code == 0:
        return _v7_gap_decode(
            data,
            count,
        )

    if code == 1:
        return (
            _v7_interpolative_decode(
                data,
                count,
                universe,
            )
        )

    if code == 2:
        return _v7_bitmap_decode(
            data,
            universe,
            count,
        )

    raise ValueError(
        "Unknown posting codec"
    )


def _v7_pack_codes(
    codes,
):
    output = bytearray(
        (
            len(codes) * 2
            + 7
        )
        // 8
    )

    for position, code in enumerate(
        codes
    ):
        bit_position = (
            position * 2
        )

        output[
            bit_position >> 3
        ] |= (
            code
            << (
                bit_position & 7
            )
        )

    return bytes(output)


def _v7_unpack_codes(
    data,
    count,
):
    output = []

    for position in range(
        count
    ):
        bit_position = (
            position * 2
        )

        output.append(
            (
                data[
                    bit_position >> 3
                ]
                >> (
                    bit_position & 7
                )
            )
            & 3
        )

    return output


def _v7_remap_collection(
    collection,
    old_to_new,
):
    output = {}

    for term, posting in (
        collection.items()
    ):
        pairs = [
            (
                old_to_new[
                    posting[position]
                ],
                posting[
                    position + 1
                ],
            )
            for position in range(
                0,
                len(posting),
                2,
            )
        ]

        pairs.sort(
            key=lambda item: item[0]
        )

        remapped = []

        append = remapped.append

        for integer_doc, tf in pairs:
            append(
                integer_doc
            )

            append(
                tf
            )

        output[
            term
        ] = remapped

    return output


def _v7_common16_reorder(
    postings,
    prefix_postings,
    doc_ids,
):
    document_count = len(
        doc_ids
    )

    if document_count <= 1:
        return (
            postings,
            prefix_postings,
            doc_ids,
        )

    terms_by_df = sorted(
        postings,
        key=lambda term: (
            -(
                len(
                    postings[
                        term
                    ]
                )
                // 2
            ),
            term,
        ),
    )

    signatures = [
        []
        for _ in range(
            document_count
        )
    ]

    for rank, term in enumerate(
        terms_by_df
    ):
        posting = postings[
            term
        ]

        for position in range(
            0,
            len(posting),
            2,
        ):
            integer_doc = posting[
                position
            ]

            signature = signatures[
                integer_doc
            ]

            if len(signature) < 16:
                signature.append(
                    rank
                )

    order = sorted(
        range(
            document_count
        ),
        key=lambda integer_doc: (
            tuple(
                signatures[
                    integer_doc
                ]
            ),
            doc_ids[
                integer_doc
            ],
        ),
    )

    old_to_new = [
        0
    ] * document_count

    new_doc_ids = [
        None
    ] * document_count

    for new_integer, old_integer in (
        enumerate(order)
    ):
        old_to_new[
            old_integer
        ] = new_integer

        new_doc_ids[
            new_integer
        ] = doc_ids[
            old_integer
        ]

    return (
        _v7_remap_collection(
            postings,
            old_to_new,
        ),
        _v7_remap_collection(
            prefix_postings,
            old_to_new,
        ),
        new_doc_ids,
    )


def _v7_build(
    self,
    corpus,
):
    postings = {}
    prefix_postings = {}

    doc_len = {}
    doc_ids = []

    total_doc_len = 0

    postings_get = postings.get
    prefix_get = (
        prefix_postings.get
    )

    for integer_doc, (
        doc_id,
        text,
    ) in enumerate(corpus):

        doc_ids.append(
            doc_id
        )

        (
            term_counts,
            prefix_counts,
            length,
        ) = _analyse_document(
            text
        )

        doc_len[
            doc_id
        ] = length

        total_doc_len += length

        for term, tf in (
            term_counts.items()
        ):
            posting = postings_get(
                term
            )

            if posting is None:
                postings[
                    term
                ] = [
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

        for term, tf in (
            prefix_counts.items()
        ):
            posting = prefix_get(
                term
            )

            if posting is None:
                prefix_postings[
                    term
                ] = [
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

    (
        postings,
        prefix_postings,
        doc_ids,
    ) = _v7_common16_reorder(
        postings,
        prefix_postings,
        doc_ids,
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


def _v7_to_integer_postings(
    postings,
    doc_to_int,
):
    output = {}

    for term, posting in (
        postings.items()
    ):
        pairs = sorted(
            (
                doc_to_int[
                    doc_id
                ],
                int(tf),
            )
            for doc_id, tf
            in posting.items()
        )

        flat = []

        append = flat.append

        for integer_doc, tf in pairs:
            append(
                integer_doc
            )

            append(
                tf
            )

        output[
            term
        ] = flat

    return output


def _v7_pack_doc_ids(
    doc_ids,
):
    if not doc_ids:
        return (
            0,
            b"",
            None,
        )

    try:
        encoded = [
            doc_id.encode(
                "ascii"
            )
            for doc_id in doc_ids
        ]
    except UnicodeEncodeError:
        return (
            0,
            b"",
            list(doc_ids),
        )

    width = len(
        encoded[0]
    )

    if (
        width > 0
        and all(
            len(value) == width
            for value in encoded
        )
    ):
        return (
            width,
            b"".join(
                encoded
            ),
            None,
        )

    return (
        0,
        b"",
        list(doc_ids),
    )


def _v7_unpack_doc_ids(
    state,
):
    width = int(
        state.get(
            "doc_id_width",
            0,
        )
    )

    if width:
        blob = state[
            "doc_id_blob"
        ]

        if len(blob) % width:
            raise ValueError(
                "Invalid document ID blob"
            )

        return [
            blob[
                position:
                position + width
            ].decode(
                "ascii"
            )
            for position in range(
                0,
                len(blob),
                width,
            )
        ]

    return list(
        state[
            "doc_ids"
        ]
    )


def _v7_save(
    self,
    index_dir,
):
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
        and len(
            possible_doc_ids
        ) == self.N
    ):
        doc_ids = list(
            possible_doc_ids
        )
    else:
        doc_ids = list(
            self.doc_len.keys()
        )

    doc_to_int = {
        doc_id: integer_doc
        for integer_doc, doc_id
        in enumerate(
            doc_ids
        )
    }

    if getattr(
        self,
        "_integer_postings",
        False,
    ):
        body_postings = (
            self.postings
        )

        prefix_postings = (
            self.prefix_postings
        )
    else:
        body_postings = (
            _v7_to_integer_postings(
                self.postings,
                doc_to_int,
            )
        )

        prefix_postings = (
            _v7_to_integer_postings(
                self.prefix_postings,
                doc_to_int,
            )
        )

    posting_terms = list(
        body_postings.keys()
    )

    body_codes = []
    body_lengths = []
    body_df = []
    body_parts = []

    body_entries = sum(
        len(posting) // 2
        for posting
        in body_postings.values()
    )

    body_tf_flags = bytearray(
        (
            body_entries + 7
        )
        // 8
    )

    body_tf_writer = (
        _V7BitWriter()
    )

    body_tf_position = 0

    for term in posting_terms:
        posting = body_postings[
            term
        ]

        docs = posting[
            0::2
        ]

        code, blob = (
            _v7_choose_codec(
                docs,
                self.N,
            )
        )

        body_codes.append(
            code
        )

        body_lengths.append(
            len(blob)
        )

        body_df.append(
            len(docs)
        )

        body_parts.append(
            blob
        )

        for position in range(
            1,
            len(posting),
            2,
        ):
            tf = int(
                posting[
                    position
                ]
            )

            if tf > 1:
                body_tf_flags[
                    body_tf_position
                    >> 3
                ] |= (
                    1
                    << (
                        body_tf_position
                        & 7
                    )
                )

                body_tf_writer.write_gamma(
                    tf - 1
                )

            body_tf_position += 1

    if body_tf_position != (
        body_entries
    ):
        raise ValueError(
            "Invalid body TF count"
        )

    prefix_bitmap = bytearray(
        (
            len(posting_terms)
            + 7
        )
        // 8
    )

    prefix_codes = []
    prefix_lengths = []
    prefix_df = []
    prefix_parts = []

    prefix_entries = sum(
        len(posting) // 2
        for posting
        in prefix_postings.values()
    )

    prefix_tf_flags = bytearray(
        (
            prefix_entries + 7
        )
        // 8
    )

    prefix_tf_writer = (
        _V7BitWriter()
    )

    prefix_tf_position = 0

    for integer_term, term in (
        enumerate(
            posting_terms
        )
    ):
        prefix = (
            prefix_postings.get(
                term
            )
        )

        if not prefix:
            continue

        prefix_bitmap[
            integer_term >> 3
        ] |= (
            1
            << (
                integer_term
                & 7
            )
        )

        body = body_postings[
            term
        ]

        positions = []

        prefix_position = 0
        prefix_length = len(
            prefix
        )

        body_position = 0

        for position in range(
            0,
            len(body),
            2,
        ):
            body_doc = body[
                position
            ]

            if (
                prefix_position
                < prefix_length
                and prefix[
                    prefix_position
                ] == body_doc
            ):
                positions.append(
                    body_position
                )

                tf = int(
                    prefix[
                        prefix_position
                        + 1
                    ]
                )

                if tf > 1:
                    prefix_tf_flags[
                        prefix_tf_position
                        >> 3
                    ] |= (
                        1
                        << (
                            prefix_tf_position
                            & 7
                        )
                    )

                    prefix_tf_writer.write_gamma(
                        tf - 1
                    )

                prefix_tf_position += 1
                prefix_position += 2

            body_position += 1

        if prefix_position != (
            prefix_length
        ):
            raise ValueError(
                "Prefix posting not contained in body posting"
            )

        code, blob = (
            _v7_choose_codec(
                positions,
                len(body) // 2,
            )
        )

        prefix_codes.append(
            code
        )

        prefix_lengths.append(
            len(blob)
        )

        prefix_df.append(
            len(positions)
        )

        prefix_parts.append(
            blob
        )

    if prefix_tf_position != (
        prefix_entries
    ):
        raise ValueError(
            "Invalid prefix TF count"
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
            self.doc_len[
                doc_id
            ]
            for doc_id
            in doc_ids
        ),
    )

    (
        doc_id_width,
        doc_id_blob,
        fallback_doc_ids,
    ) = _v7_pack_doc_ids(
        doc_ids
    )

    state = {
        "format_version": 7,
        "doc_id_width":
            doc_id_width,
        "doc_id_blob":
            doc_id_blob,
        "doc_lengths":
            doc_lengths,
        "posting_terms":
            posting_terms,
        "body_codec_flags":
            _v7_pack_codes(
                body_codes
            ),
        "body_lengths_blob":
            _v7_encode_varints(
                body_lengths
            ),
        "body_df_blob":
            _v7_encode_varints(
                body_df
            ),
        "body_blob":
            b"".join(
                body_parts
            ),
        "body_tf_flags":
            bytes(
                body_tf_flags
            ),
        "body_tf_blob":
            body_tf_writer.finish(),
        "prefix_bitmap":
            bytes(
                prefix_bitmap
            ),
        "prefix_codec_flags":
            _v7_pack_codes(
                prefix_codes
            ),
        "prefix_lengths_blob":
            _v7_encode_varints(
                prefix_lengths
            ),
        "prefix_df_blob":
            _v7_encode_varints(
                prefix_df
            ),
        "prefix_blob":
            b"".join(
                prefix_parts
            ),
        "prefix_tf_flags":
            bytes(
                prefix_tf_flags
            ),
        "prefix_tf_blob":
            prefix_tf_writer.finish(),
        "N":
            self.N,
        "avg_doc_len":
            self.avg_doc_len,
    }

    if fallback_doc_ids is not None:
        state[
            "doc_ids"
        ] = fallback_doc_ids

    path = os.path.join(
        index_dir,
        _INDEX_FILENAME,
    )

    with bz2.open(
        path,
        "wb",
        compresslevel=9,
    ) as f:
        pickle.dump(
            state,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _v7_load(
    cls,
    index_dir,
):
    path = os.path.join(
        index_dir,
        _INDEX_FILENAME,
    )

    with bz2.open(
        path,
        "rb",
    ) as f:
        state = pickle.load(
            f
        )

    if state.get(
        "format_version",
        2,
    ) < 7:
        return _V7_LEGACY_LOAD(
            index_dir
        )

    doc_ids = (
        _v7_unpack_doc_ids(
            state
        )
    )

    doc_lengths = state[
        "doc_lengths"
    ]

    if len(doc_ids) != len(
        doc_lengths
    ):
        raise ValueError(
            "Document metadata mismatch"
        )

    posting_terms = state[
        "posting_terms"
    ]

    term_count = len(
        posting_terms
    )

    body_codes = (
        _v7_unpack_codes(
            state[
                "body_codec_flags"
            ],
            term_count,
        )
    )

    body_lengths = (
        _v7_decode_varints(
            state[
                "body_lengths_blob"
            ],
            term_count,
        )
    )

    body_df = (
        _v7_decode_varints(
            state[
                "body_df_blob"
            ],
            term_count,
        )
    )

    body_blob = state[
        "body_blob"
    ]

    body_tf_flags = state[
        "body_tf_flags"
    ]

    body_tf_reader = (
        _V7BitReader(
            state[
                "body_tf_blob"
            ]
        )
    )

    postings = {}

    body_blob_position = 0
    body_tf_position = 0

    for (
        term,
        code,
        length,
        df,
    ) in zip(
        posting_terms,
        body_codes,
        body_lengths,
        body_df,
    ):
        next_position = (
            body_blob_position
            + length
        )

        data = body_blob[
            body_blob_position:
            next_position
        ]

        docs = _v7_decode_codec(
            code,
            data,
            df,
            state[
                "N"
            ],
        )

        posting = {}

        for integer_doc in docs:
            tf = 1

            if body_tf_flags[
                body_tf_position
                >> 3
            ] & (
                1
                << (
                    body_tf_position
                    & 7
                )
            ):
                tf = (
                    body_tf_reader.read_gamma()
                    + 1
                )

            posting[
                doc_ids[
                    integer_doc
                ]
            ] = tf

            body_tf_position += 1

        postings[
            term
        ] = posting

        body_blob_position = (
            next_position
        )

    if body_blob_position != len(
        body_blob
    ):
        raise ValueError(
            "Invalid body blob"
        )

    prefix_bitmap = state[
        "prefix_bitmap"
    ]

    prefix_term_count = sum(
        int(byte).bit_count()
        for byte
        in prefix_bitmap
    )

    prefix_codes = (
        _v7_unpack_codes(
            state[
                "prefix_codec_flags"
            ],
            prefix_term_count,
        )
    )

    prefix_lengths = (
        _v7_decode_varints(
            state[
                "prefix_lengths_blob"
            ],
            prefix_term_count,
        )
    )

    prefix_df = (
        _v7_decode_varints(
            state[
                "prefix_df_blob"
            ],
            prefix_term_count,
        )
    )

    prefix_blob = state[
        "prefix_blob"
    ]

    prefix_tf_flags = state[
        "prefix_tf_flags"
    ]

    prefix_tf_reader = (
        _V7BitReader(
            state[
                "prefix_tf_blob"
            ]
        )
    )

    prefix_postings = {}

    prefix_blob_position = 0
    prefix_metadata_position = 0
    prefix_tf_position = 0

    for integer_term, term in (
        enumerate(
            posting_terms
        )
    ):
        if not (
            prefix_bitmap[
                integer_term >> 3
            ]
            & (
                1
                << (
                    integer_term
                    & 7
                )
            )
        ):
            continue

        code = prefix_codes[
            prefix_metadata_position
        ]

        length = prefix_lengths[
            prefix_metadata_position
        ]

        df = prefix_df[
            prefix_metadata_position
        ]

        next_position = (
            prefix_blob_position
            + length
        )

        data = prefix_blob[
            prefix_blob_position:
            next_position
        ]

        body_doc_ids = list(
            postings[
                term
            ].keys()
        )

        positions = (
            _v7_decode_codec(
                code,
                data,
                df,
                len(
                    body_doc_ids
                ),
            )
        )

        posting = {}

        for body_position in positions:
            tf = 1

            if prefix_tf_flags[
                prefix_tf_position
                >> 3
            ] & (
                1
                << (
                    prefix_tf_position
                    & 7
                )
            ):
                tf = (
                    prefix_tf_reader.read_gamma()
                    + 1
                )

            posting[
                body_doc_ids[
                    body_position
                ]
            ] = tf

            prefix_tf_position += 1

        prefix_postings[
            term
        ] = posting

        prefix_blob_position = (
            next_position
        )

        prefix_metadata_position += 1

    if prefix_blob_position != len(
        prefix_blob
    ):
        raise ValueError(
            "Invalid prefix blob"
        )

    if prefix_metadata_position != (
        prefix_term_count
    ):
        raise ValueError(
            "Invalid prefix metadata"
        )

    index = cls()

    index.postings = postings

    index.prefix_postings = (
        prefix_postings
    )

    index.doc_len = {
        doc_id: int(length)
        for doc_id, length
        in zip(
            doc_ids,
            doc_lengths,
        )
    }

    index.doc_ids = doc_ids

    index._integer_postings = (
        False
    )

    index.N = int(
        state[
            "N"
        ]
    )

    index.avg_doc_len = float(
        state[
            "avg_doc_len"
        ]
    )

    index.doc_text = {}

    return index


InvertedIndex.build = _v7_build

InvertedIndex.save = _v7_save

InvertedIndex.load = classmethod(
    _v7_load
)

def _v8_build(
    self,
    corpus,
):
    term_to_int = {}
    terms = []
    document_frequencies = []

    body_term_ids = array(
        "I"
    )

    body_tfs = array(
        "I"
    )

    body_offsets = array(
        "I",
        [0],
    )

    prefix_term_ids = array(
        "I"
    )

    prefix_tfs = array(
        "I"
    )

    prefix_offsets = array(
        "I",
        [0],
    )

    doc_ids = []
    doc_len = {}

    total_doc_len = 0

    term_get = term_to_int.get
    terms_append = terms.append
    df_append = (
        document_frequencies.append
    )

    body_term_append = (
        body_term_ids.append
    )

    body_tf_append = (
        body_tfs.append
    )

    body_offset_append = (
        body_offsets.append
    )

    prefix_term_append = (
        prefix_term_ids.append
    )

    prefix_tf_append = (
        prefix_tfs.append
    )

    prefix_offset_append = (
        prefix_offsets.append
    )

    for doc_id, text in corpus:
        doc_ids.append(
            doc_id
        )

        (
            term_counts,
            prefix_counts,
            length,
        ) = _analyse_document(
            text
        )

        doc_len[
            doc_id
        ] = length

        total_doc_len += length

        for term, tf in (
            term_counts.items()
        ):
            integer_term = term_get(
                term
            )

            if integer_term is None:
                integer_term = len(
                    terms
                )

                term_to_int[
                    term
                ] = integer_term

                terms_append(
                    term
                )

                df_append(
                    0
                )

            document_frequencies[
                integer_term
            ] += 1

            body_term_append(
                integer_term
            )

            body_tf_append(
                int(tf)
            )

        body_offset_append(
            len(
                body_term_ids
            )
        )

        for term, tf in (
            prefix_counts.items()
        ):
            integer_term = term_get(
                term
            )

            if integer_term is None:
                raise ValueError(
                    "Prefix term missing from body terms"
                )

            prefix_term_append(
                integer_term
            )

            prefix_tf_append(
                int(tf)
            )

        prefix_offset_append(
            len(
                prefix_term_ids
            )
        )

    document_count = len(
        doc_ids
    )

    term_count = len(
        terms
    )

    ranked_terms = sorted(
        range(
            term_count
        ),
        key=lambda integer_term: (
            -document_frequencies[
                integer_term
            ],
            terms[
                integer_term
            ],
        ),
    )

    term_rank = [
        0
    ] * term_count

    for rank, integer_term in (
        enumerate(
            ranked_terms
        )
    ):
        term_rank[
            integer_term
        ] = rank

    signatures = [
        None
    ] * document_count

    body_term_ids_local = (
        body_term_ids
    )

    body_offsets_local = (
        body_offsets
    )

    term_rank_local = (
        term_rank
    )

    for integer_doc in range(
        document_count
    ):
        start = body_offsets_local[
            integer_doc
        ]

        end = body_offsets_local[
            integer_doc + 1
        ]

        ranks = [
            term_rank_local[
                body_term_ids_local[
                    position
                ]
            ]
            for position in range(
                start,
                end,
            )
        ]

        ranks.sort()

        signatures[
            integer_doc
        ] = tuple(
            ranks[
                :16
            ]
        )

    order = sorted(
        range(
            document_count
        ),
        key=lambda integer_doc: (
            signatures[
                integer_doc
            ],
            doc_ids[
                integer_doc
            ],
        ),
    )

    body_lists = [
        []
        for _ in range(
            term_count
        )
    ]

    prefix_lists = [
        None
        for _ in range(
            term_count
        )
    ]

    new_doc_ids = [
        None
    ] * document_count

    body_tfs_local = (
        body_tfs
    )

    prefix_term_ids_local = (
        prefix_term_ids
    )

    prefix_tfs_local = (
        prefix_tfs
    )

    prefix_offsets_local = (
        prefix_offsets
    )

    for new_integer_doc, (
        old_integer_doc
    ) in enumerate(
        order
    ):
        new_doc_ids[
            new_integer_doc
        ] = doc_ids[
            old_integer_doc
        ]

        start = body_offsets_local[
            old_integer_doc
        ]

        end = body_offsets_local[
            old_integer_doc + 1
        ]

        for position in range(
            start,
            end,
        ):
            integer_term = (
                body_term_ids_local[
                    position
                ]
            )

            posting = body_lists[
                integer_term
            ]

            posting.append(
                new_integer_doc
            )

            posting.append(
                int(
                    body_tfs_local[
                        position
                    ]
                )
            )

        start = prefix_offsets_local[
            old_integer_doc
        ]

        end = prefix_offsets_local[
            old_integer_doc + 1
        ]

        for position in range(
            start,
            end,
        ):
            integer_term = (
                prefix_term_ids_local[
                    position
                ]
            )

            posting = prefix_lists[
                integer_term
            ]

            if posting is None:
                posting = []

                prefix_lists[
                    integer_term
                ] = posting

            posting.append(
                new_integer_doc
            )

            posting.append(
                int(
                    prefix_tfs_local[
                        position
                    ]
                )
            )

    postings = {
        terms[
            integer_term
        ]:
        body_lists[
            integer_term
        ]
        for integer_term in range(
            term_count
        )
        if body_lists[
            integer_term
        ]
    }

    prefix_postings = {
        terms[
            integer_term
        ]:
        prefix_lists[
            integer_term
        ]
        for integer_term in range(
            term_count
        )
        if prefix_lists[
            integer_term
        ]
    }

    self.postings = postings

    self.prefix_postings = (
        prefix_postings
    )

    self.doc_len = doc_len

    self.doc_ids = new_doc_ids

    self._integer_postings = True

    self.doc_text = {}

    self.N = document_count

    if document_count:
        self.avg_doc_len = (
            total_doc_len
            / document_count
        )
    else:
        self.avg_doc_len = 0.0


InvertedIndex.build = _v8_build

_V9_PREVIOUS_LOAD = InvertedIndex.load


def _v9_write_interpolative(
    writer,
    values,
    universe,
):
    if not values:
        return

    stack = [
        (
            0,
            len(values) - 1,
            0,
            universe - 1,
        )
    ]

    while stack:
        (
            left,
            right,
            low,
            high,
        ) = stack.pop()

        if left > right:
            continue

        middle = (
            left + right
        ) // 2

        minimum = (
            low
            + middle
            - left
        )

        maximum = (
            high
            - (
                right - middle
            )
        )

        count = (
            maximum
            - minimum
            + 1
        )

        middle_value = values[
            middle
        ]

        writer.write_bounded(
            middle_value
            - minimum,
            count,
        )

        stack.append(
            (
                middle + 1,
                right,
                middle_value + 1,
                high,
            )
        )

        stack.append(
            (
                left,
                middle - 1,
                low,
                middle_value - 1,
            )
        )


def _v9_read_interpolative(
    reader,
    count,
    universe,
):
    if count == 0:
        return []

    values = [
        0
    ] * count

    stack = [
        (
            0,
            count - 1,
            0,
            universe - 1,
        )
    ]

    while stack:
        (
            left,
            right,
            low,
            high,
        ) = stack.pop()

        if left > right:
            continue

        middle = (
            left + right
        ) // 2

        minimum = (
            low
            + middle
            - left
        )

        maximum = (
            high
            - (
                right - middle
            )
        )

        possible = (
            maximum
            - minimum
            + 1
        )

        middle_value = (
            minimum
            + reader.read_bounded(
                possible
            )
        )

        values[
            middle
        ] = middle_value

        stack.append(
            (
                middle + 1,
                right,
                middle_value + 1,
                high,
            )
        )

        stack.append(
            (
                left,
                middle - 1,
                low,
                middle_value - 1,
            )
        )

    return values


def _v9_save(
    self,
    index_dir,
):
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
        and len(
            possible_doc_ids
        ) == self.N
    ):
        doc_ids = list(
            possible_doc_ids
        )
    else:
        doc_ids = list(
            self.doc_len.keys()
        )

    doc_to_int = {
        doc_id: integer_doc
        for integer_doc, doc_id
        in enumerate(
            doc_ids
        )
    }

    if getattr(
        self,
        "_integer_postings",
        False,
    ):
        body_postings = (
            self.postings
        )

        prefix_postings = (
            self.prefix_postings
        )
    else:
        body_postings = (
            _v7_to_integer_postings(
                self.postings,
                doc_to_int,
            )
        )

        prefix_postings = (
            _v7_to_integer_postings(
                self.prefix_postings,
                doc_to_int,
            )
        )

    posting_terms = list(
        body_postings.keys()
    )

    body_df = []

    body_entries = sum(
        len(posting) // 2
        for posting
        in body_postings.values()
    )

    body_doc_writer = (
        _V7BitWriter()
    )

    body_tf_flags = bytearray(
        (
            body_entries + 7
        )
        // 8
    )

    body_tf_writer = (
        _V7BitWriter()
    )

    body_tf_position = 0

    for term in posting_terms:
        posting = body_postings[
            term
        ]

        docs = posting[
            0::2
        ]

        body_df.append(
            len(docs)
        )

        _v9_write_interpolative(
            body_doc_writer,
            docs,
            self.N,
        )

        for position in range(
            1,
            len(posting),
            2,
        ):
            tf = int(
                posting[
                    position
                ]
            )

            if tf > 1:
                body_tf_flags[
                    body_tf_position
                    >> 3
                ] |= (
                    1
                    << (
                        body_tf_position
                        & 7
                    )
                )

                body_tf_writer.write_gamma(
                    tf - 1
                )

            body_tf_position += 1

    if body_tf_position != (
        body_entries
    ):
        raise ValueError(
            "Invalid body TF count"
        )

    prefix_bitmap = bytearray(
        (
            len(posting_terms)
            + 7
        )
        // 8
    )

    prefix_df = []

    prefix_entries = sum(
        len(posting) // 2
        for posting
        in prefix_postings.values()
    )

    prefix_position_writer = (
        _V7BitWriter()
    )

    prefix_tf_flags = bytearray(
        (
            prefix_entries + 7
        )
        // 8
    )

    prefix_tf_writer = (
        _V7BitWriter()
    )

    prefix_tf_position = 0

    for integer_term, term in (
        enumerate(
            posting_terms
        )
    ):
        prefix = (
            prefix_postings.get(
                term
            )
        )

        if not prefix:
            continue

        prefix_bitmap[
            integer_term >> 3
        ] |= (
            1
            << (
                integer_term
                & 7
            )
        )

        body = body_postings[
            term
        ]

        positions = []

        prefix_position = 0
        prefix_length = len(
            prefix
        )

        body_position = 0

        for position in range(
            0,
            len(body),
            2,
        ):
            body_doc = body[
                position
            ]

            if (
                prefix_position
                < prefix_length
                and prefix[
                    prefix_position
                ] == body_doc
            ):
                positions.append(
                    body_position
                )

                tf = int(
                    prefix[
                        prefix_position
                        + 1
                    ]
                )

                if tf > 1:
                    prefix_tf_flags[
                        prefix_tf_position
                        >> 3
                    ] |= (
                        1
                        << (
                            prefix_tf_position
                            & 7
                        )
                    )

                    prefix_tf_writer.write_gamma(
                        tf - 1
                    )

                prefix_tf_position += 1
                prefix_position += 2

            body_position += 1

        if prefix_position != (
            prefix_length
        ):
            raise ValueError(
                "Prefix posting not contained in body posting"
            )

        prefix_df.append(
            len(positions)
        )

        _v9_write_interpolative(
            prefix_position_writer,
            positions,
            len(body) // 2,
        )

    if prefix_tf_position != (
        prefix_entries
    ):
        raise ValueError(
            "Invalid prefix TF count"
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
            self.doc_len[
                doc_id
            ]
            for doc_id
            in doc_ids
        ),
    )

    (
        doc_id_width,
        doc_id_blob,
        fallback_doc_ids,
    ) = _v7_pack_doc_ids(
        doc_ids
    )

    state = {
        "format_version": 9,
        "doc_id_width":
            doc_id_width,
        "doc_id_blob":
            doc_id_blob,
        "doc_lengths":
            doc_lengths,
        "posting_terms":
            posting_terms,
        "body_df_blob":
            _v7_encode_varints(
                body_df
            ),
        "body_doc_blob":
            body_doc_writer.finish(),
        "body_tf_flags":
            bytes(
                body_tf_flags
            ),
        "body_tf_blob":
            body_tf_writer.finish(),
        "prefix_bitmap":
            bytes(
                prefix_bitmap
            ),
        "prefix_df_blob":
            _v7_encode_varints(
                prefix_df
            ),
        "prefix_position_blob":
            prefix_position_writer.finish(),
        "prefix_tf_flags":
            bytes(
                prefix_tf_flags
            ),
        "prefix_tf_blob":
            prefix_tf_writer.finish(),
        "N":
            self.N,
        "avg_doc_len":
            self.avg_doc_len,
    }

    if fallback_doc_ids is not None:
        state[
            "doc_ids"
        ] = fallback_doc_ids

    path = os.path.join(
        index_dir,
        _INDEX_FILENAME,
    )

    with bz2.open(
        path,
        "wb",
        compresslevel=9,
    ) as f:
        pickle.dump(
            state,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _v9_load(
    cls,
    index_dir,
):
    path = os.path.join(
        index_dir,
        _INDEX_FILENAME,
    )

    with bz2.open(
        path,
        "rb",
    ) as f:
        state = pickle.load(
            f
        )

    if state.get(
        "format_version",
        2,
    ) < 9:
        return _V9_PREVIOUS_LOAD(
            index_dir
        )

    doc_ids = (
        _v7_unpack_doc_ids(
            state
        )
    )

    doc_lengths = state[
        "doc_lengths"
    ]

    posting_terms = state[
        "posting_terms"
    ]

    term_count = len(
        posting_terms
    )

    body_df = (
        _v7_decode_varints(
            state[
                "body_df_blob"
            ],
            term_count,
        )
    )

    body_doc_reader = (
        _V7BitReader(
            state[
                "body_doc_blob"
            ]
        )
    )

    body_tf_flags = state[
        "body_tf_flags"
    ]

    body_tf_reader = (
        _V7BitReader(
            state[
                "body_tf_blob"
            ]
        )
    )

    postings = {}

    body_tf_position = 0

    for term, df in zip(
        posting_terms,
        body_df,
    ):
        docs = (
            _v9_read_interpolative(
                body_doc_reader,
                df,
                state[
                    "N"
                ],
            )
        )

        posting = {}

        for integer_doc in docs:
            tf = 1

            if body_tf_flags[
                body_tf_position
                >> 3
            ] & (
                1
                << (
                    body_tf_position
                    & 7
                )
            ):
                tf = (
                    body_tf_reader.read_gamma()
                    + 1
                )

            posting[
                doc_ids[
                    integer_doc
                ]
            ] = tf

            body_tf_position += 1

        postings[
            term
        ] = posting

    prefix_bitmap = state[
        "prefix_bitmap"
    ]

    prefix_term_count = sum(
        int(byte).bit_count()
        for byte
        in prefix_bitmap
    )

    prefix_df = (
        _v7_decode_varints(
            state[
                "prefix_df_blob"
            ],
            prefix_term_count,
        )
    )

    prefix_position_reader = (
        _V7BitReader(
            state[
                "prefix_position_blob"
            ]
        )
    )

    prefix_tf_flags = state[
        "prefix_tf_flags"
    ]

    prefix_tf_reader = (
        _V7BitReader(
            state[
                "prefix_tf_blob"
            ]
        )
    )

    prefix_postings = {}

    prefix_metadata_position = 0
    prefix_tf_position = 0

    for integer_term, term in (
        enumerate(
            posting_terms
        )
    ):
        if not (
            prefix_bitmap[
                integer_term >> 3
            ]
            & (
                1
                << (
                    integer_term
                    & 7
                )
            )
        ):
            continue

        df = prefix_df[
            prefix_metadata_position
        ]

        body_doc_ids = list(
            postings[
                term
            ].keys()
        )

        positions = (
            _v9_read_interpolative(
                prefix_position_reader,
                df,
                len(
                    body_doc_ids
                ),
            )
        )

        posting = {}

        for body_position in positions:
            tf = 1

            if prefix_tf_flags[
                prefix_tf_position
                >> 3
            ] & (
                1
                << (
                    prefix_tf_position
                    & 7
                )
            ):
                tf = (
                    prefix_tf_reader.read_gamma()
                    + 1
                )

            posting[
                body_doc_ids[
                    body_position
                ]
            ] = tf

            prefix_tf_position += 1

        prefix_postings[
            term
        ] = posting

        prefix_metadata_position += 1

    if prefix_metadata_position != (
        prefix_term_count
    ):
        raise ValueError(
            "Invalid prefix metadata"
        )

    index = cls()

    index.postings = postings

    index.prefix_postings = (
        prefix_postings
    )

    index.doc_len = {
        doc_id: int(length)
        for doc_id, length
        in zip(
            doc_ids,
            doc_lengths,
        )
    }

    index.doc_ids = doc_ids

    index._integer_postings = False

    index.N = int(
        state[
            "N"
        ]
    )

    index.avg_doc_len = float(
        state[
            "avg_doc_len"
        ]
    )

    index.doc_text = {}

    return index


InvertedIndex.save = _v9_save

InvertedIndex.load = classmethod(
    _v9_load
)

_V10_PREVIOUS_LOAD = InvertedIndex.load


class _V10VarintReader:

    def __init__(
        self,
        data,
    ):
        self.data = data
        self.position = 0

    def read(self):
        value = 0
        shift = 0
        data = self.data
        position = self.position

        while True:
            byte = data[position]
            position += 1

            value |= (
                byte & 127
            ) << shift

            if byte < 128:
                break

            shift += 7

        self.position = position

        return value


def _v10_save(
    self,
    index_dir,
):
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

    doc_to_int = {
        doc_id: integer_doc
        for integer_doc, doc_id
        in enumerate(doc_ids)
    }

    if getattr(
        self,
        "_integer_postings",
        False,
    ):
        body_postings = self.postings
        prefix_postings = (
            self.prefix_postings
        )
    else:
        body_postings = (
            _v7_to_integer_postings(
                self.postings,
                doc_to_int,
            )
        )

        prefix_postings = (
            _v7_to_integer_postings(
                self.prefix_postings,
                doc_to_int,
            )
        )

    posting_terms = list(
        body_postings.keys()
    )

    body_df = []

    body_entries = sum(
        len(posting) // 2
        for posting
        in body_postings.values()
    )

    body_doc_blob = bytearray()

    body_tf_flags = bytearray(
        (
            body_entries + 7
        )
        // 8
    )

    body_tf_writer = (
        _V7BitWriter()
    )

    body_tf_position = 0

    for term in posting_terms:
        posting = body_postings[
            term
        ]

        df = len(posting) // 2

        body_df.append(
            df
        )

        previous_doc = -1

        for position in range(
            0,
            len(posting),
            2,
        ):
            integer_doc = int(
                posting[
                    position
                ]
            )

            tf = int(
                posting[
                    position + 1
                ]
            )

            _write_varint(
                body_doc_blob,
                integer_doc
                - previous_doc,
            )

            previous_doc = (
                integer_doc
            )

            if tf > 1:
                body_tf_flags[
                    body_tf_position
                    >> 3
                ] |= (
                    1
                    << (
                        body_tf_position
                        & 7
                    )
                )

                body_tf_writer.write_gamma(
                    tf - 1
                )

            body_tf_position += 1

    if body_tf_position != (
        body_entries
    ):
        raise ValueError(
            "Invalid body TF count"
        )

    prefix_bitmap = bytearray(
        (
            len(posting_terms)
            + 7
        )
        // 8
    )

    prefix_df = []

    prefix_entries = sum(
        len(posting) // 2
        for posting
        in prefix_postings.values()
    )

    prefix_position_blob = (
        bytearray()
    )

    prefix_tf_flags = bytearray(
        (
            prefix_entries + 7
        )
        // 8
    )

    prefix_tf_writer = (
        _V7BitWriter()
    )

    prefix_tf_position = 0

    for integer_term, term in (
        enumerate(
            posting_terms
        )
    ):
        prefix = (
            prefix_postings.get(
                term
            )
        )

        if not prefix:
            continue

        prefix_bitmap[
            integer_term >> 3
        ] |= (
            1
            << (
                integer_term
                & 7
            )
        )

        body = body_postings[
            term
        ]

        df = len(prefix) // 2

        prefix_df.append(
            df
        )

        prefix_position = 0
        prefix_length = len(
            prefix
        )

        previous_body_position = -1
        body_position = 0

        for position in range(
            0,
            len(body),
            2,
        ):
            body_doc = body[
                position
            ]

            if (
                prefix_position
                < prefix_length
                and prefix[
                    prefix_position
                ] == body_doc
            ):
                _write_varint(
                    prefix_position_blob,
                    body_position
                    - previous_body_position,
                )

                previous_body_position = (
                    body_position
                )

                tf = int(
                    prefix[
                        prefix_position
                        + 1
                    ]
                )

                if tf > 1:
                    prefix_tf_flags[
                        prefix_tf_position
                        >> 3
                    ] |= (
                        1
                        << (
                            prefix_tf_position
                            & 7
                        )
                    )

                    prefix_tf_writer.write_gamma(
                        tf - 1
                    )

                prefix_tf_position += 1
                prefix_position += 2

            body_position += 1

        if prefix_position != (
            prefix_length
        ):
            raise ValueError(
                "Prefix posting not contained in body posting"
            )

    if prefix_tf_position != (
        prefix_entries
    ):
        raise ValueError(
            "Invalid prefix TF count"
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
            self.doc_len[
                doc_id
            ]
            for doc_id in doc_ids
        ),
    )

    (
        doc_id_width,
        doc_id_blob,
        fallback_doc_ids,
    ) = _v7_pack_doc_ids(
        doc_ids
    )

    state = {
        "format_version": 10,
        "doc_id_width":
            doc_id_width,
        "doc_id_blob":
            doc_id_blob,
        "doc_lengths":
            doc_lengths,
        "posting_terms":
            posting_terms,
        "body_df_blob":
            _v7_encode_varints(
                body_df
            ),
        "body_doc_blob":
            bytes(
                body_doc_blob
            ),
        "body_tf_flags":
            bytes(
                body_tf_flags
            ),
        "body_tf_blob":
            body_tf_writer.finish(),
        "prefix_bitmap":
            bytes(
                prefix_bitmap
            ),
        "prefix_df_blob":
            _v7_encode_varints(
                prefix_df
            ),
        "prefix_position_blob":
            bytes(
                prefix_position_blob
            ),
        "prefix_tf_flags":
            bytes(
                prefix_tf_flags
            ),
        "prefix_tf_blob":
            prefix_tf_writer.finish(),
        "N":
            self.N,
        "avg_doc_len":
            self.avg_doc_len,
    }

    if fallback_doc_ids is not None:
        state[
            "doc_ids"
        ] = fallback_doc_ids

    path = os.path.join(
        index_dir,
        _INDEX_FILENAME,
    )

    with bz2.open(
        path,
        "wb",
        compresslevel=9,
    ) as f:
        pickle.dump(
            state,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _v10_load(
    cls,
    index_dir,
):
    path = os.path.join(
        index_dir,
        _INDEX_FILENAME,
    )

    with bz2.open(
        path,
        "rb",
    ) as f:
        state = pickle.load(
            f
        )

    if state.get(
        "format_version",
        2,
    ) < 10:
        return _V10_PREVIOUS_LOAD(
            index_dir
        )

    doc_ids = (
        _v7_unpack_doc_ids(
            state
        )
    )

    doc_lengths = state[
        "doc_lengths"
    ]

    posting_terms = state[
        "posting_terms"
    ]

    term_count = len(
        posting_terms
    )

    body_df = (
        _v7_decode_varints(
            state[
                "body_df_blob"
            ],
            term_count,
        )
    )

    body_reader = (
        _V10VarintReader(
            state[
                "body_doc_blob"
            ]
        )
    )

    body_tf_flags = state[
        "body_tf_flags"
    ]

    body_tf_reader = (
        _V7BitReader(
            state[
                "body_tf_blob"
            ]
        )
    )

    postings = {}

    body_tf_position = 0

    for term, df in zip(
        posting_terms,
        body_df,
    ):
        posting = {}

        previous_doc = -1

        for _ in range(
            df
        ):
            previous_doc += (
                body_reader.read()
            )

            tf = 1

            if body_tf_flags[
                body_tf_position
                >> 3
            ] & (
                1
                << (
                    body_tf_position
                    & 7
                )
            ):
                tf = (
                    body_tf_reader.read_gamma()
                    + 1
                )

            posting[
                doc_ids[
                    previous_doc
                ]
            ] = tf

            body_tf_position += 1

        postings[
            term
        ] = posting

    if body_reader.position != len(
        state[
            "body_doc_blob"
        ]
    ):
        raise ValueError(
            "Invalid body document stream"
        )

    prefix_bitmap = state[
        "prefix_bitmap"
    ]

    prefix_term_count = sum(
        int(byte).bit_count()
        for byte
        in prefix_bitmap
    )

    prefix_df = (
        _v7_decode_varints(
            state[
                "prefix_df_blob"
            ],
            prefix_term_count,
        )
    )

    prefix_reader = (
        _V10VarintReader(
            state[
                "prefix_position_blob"
            ]
        )
    )

    prefix_tf_flags = state[
        "prefix_tf_flags"
    ]

    prefix_tf_reader = (
        _V7BitReader(
            state[
                "prefix_tf_blob"
            ]
        )
    )

    prefix_postings = {}

    prefix_metadata_position = 0
    prefix_tf_position = 0

    for integer_term, term in (
        enumerate(
            posting_terms
        )
    ):
        if not (
            prefix_bitmap[
                integer_term >> 3
            ]
            & (
                1
                << (
                    integer_term
                    & 7
                )
            )
        ):
            continue

        df = prefix_df[
            prefix_metadata_position
        ]

        body_doc_ids = list(
            postings[
                term
            ].keys()
        )

        posting = {}

        previous_position = -1

        for _ in range(
            df
        ):
            previous_position += (
                prefix_reader.read()
            )

            tf = 1

            if prefix_tf_flags[
                prefix_tf_position
                >> 3
            ] & (
                1
                << (
                    prefix_tf_position
                    & 7
                )
            ):
                tf = (
                    prefix_tf_reader.read_gamma()
                    + 1
                )

            posting[
                body_doc_ids[
                    previous_position
                ]
            ] = tf

            prefix_tf_position += 1

        prefix_postings[
            term
        ] = posting

        prefix_metadata_position += 1

    if prefix_reader.position != len(
        state[
            "prefix_position_blob"
        ]
    ):
        raise ValueError(
            "Invalid prefix position stream"
        )

    if prefix_metadata_position != (
        prefix_term_count
    ):
        raise ValueError(
            "Invalid prefix metadata"
        )

    index = cls()

    index.postings = postings

    index.prefix_postings = (
        prefix_postings
    )

    index.doc_len = {
        doc_id: int(length)
        for doc_id, length
        in zip(
            doc_ids,
            doc_lengths,
        )
    }

    index.doc_ids = doc_ids

    index._integer_postings = False

    index.N = int(
        state[
            "N"
        ]
    )

    index.avg_doc_len = float(
        state[
            "avg_doc_len"
        ]
    )

    index.doc_text = {}

    return index


InvertedIndex.save = _v10_save

InvertedIndex.load = classmethod(
    _v10_load
)

_V12_CORPUS = None


def _v12_analyse_at(
    integer_doc,
):
    text = _V12_CORPUS[
        integer_doc
    ][1]

    (
        term_counts,
        prefix_counts,
        length,
    ) = _analyse_document(
        text
    )

    return (
        integer_doc,
        term_counts,
        prefix_counts,
        length,
    )


def _v12_build(
    self,
    corpus,
):
    import multiprocessing as mp

    global _V12_CORPUS

    term_to_int = {}
    terms = []
    document_frequencies = []

    body_term_ids = array(
        "I"
    )

    body_tfs = array(
        "I"
    )

    body_offsets = array(
        "I",
        [0],
    )

    prefix_term_ids = array(
        "I"
    )

    prefix_tfs = array(
        "I"
    )

    prefix_offsets = array(
        "I",
        [0],
    )

    doc_ids = []
    doc_len = {}

    total_doc_len = 0

    term_get = term_to_int.get
    terms_append = terms.append
    df_append = document_frequencies.append

    body_term_append = body_term_ids.append
    body_tf_append = body_tfs.append
    body_offset_append = body_offsets.append

    prefix_term_append = prefix_term_ids.append
    prefix_tf_append = prefix_tfs.append
    prefix_offset_append = prefix_offsets.append

    document_count = len(
        corpus
    )

    workers = min(
        4,
        os.cpu_count() or 1,
    )

    _V12_CORPUS = corpus

    context = mp.get_context(
        "fork"
    )

    try:
        with context.Pool(
            processes=workers
        ) as pool:
            results = pool.imap(
                _v12_analyse_at,
                range(
                    document_count
                ),
                chunksize=512,
            )

            for (
                integer_doc,
                term_counts,
                prefix_counts,
                length,
            ) in results:
                doc_id = corpus[
                    integer_doc
                ][0]

                doc_ids.append(
                    doc_id
                )

                doc_len[
                    doc_id
                ] = length

                total_doc_len += length

                for term, tf in (
                    term_counts.items()
                ):
                    integer_term = term_get(
                        term
                    )

                    if integer_term is None:
                        integer_term = len(
                            terms
                        )

                        term_to_int[
                            term
                        ] = integer_term

                        terms_append(
                            term
                        )

                        df_append(
                            0
                        )

                    document_frequencies[
                        integer_term
                    ] += 1

                    body_term_append(
                        integer_term
                    )

                    body_tf_append(
                        int(
                            tf
                        )
                    )

                body_offset_append(
                    len(
                        body_term_ids
                    )
                )

                for term, tf in (
                    prefix_counts.items()
                ):
                    integer_term = term_get(
                        term
                    )

                    if integer_term is None:
                        raise ValueError(
                            "Prefix term missing from body terms"
                        )

                    prefix_term_append(
                        integer_term
                    )

                    prefix_tf_append(
                        int(
                            tf
                        )
                    )

                prefix_offset_append(
                    len(
                        prefix_term_ids
                    )
                )
    finally:
        _V12_CORPUS = None

    term_count = len(
        terms
    )

    ranked_terms = sorted(
        range(
            term_count
        ),
        key=lambda integer_term: (
            -document_frequencies[
                integer_term
            ],
            terms[
                integer_term
            ],
        ),
    )

    term_rank = [
        0
    ] * term_count

    for rank, integer_term in enumerate(
        ranked_terms
    ):
        term_rank[
            integer_term
        ] = rank

    signatures = [
        None
    ] * document_count

    body_term_ids_local = body_term_ids
    body_offsets_local = body_offsets
    term_rank_local = term_rank

    for integer_doc in range(
        document_count
    ):
        start = body_offsets_local[
            integer_doc
        ]

        end = body_offsets_local[
            integer_doc + 1
        ]

        ranks = [
            term_rank_local[
                body_term_ids_local[
                    position
                ]
            ]
            for position in range(
                start,
                end,
            )
        ]

        ranks.sort()

        signatures[
            integer_doc
        ] = tuple(
            ranks[
                :16
            ]
        )

    order = sorted(
        range(
            document_count
        ),
        key=lambda integer_doc: (
            signatures[
                integer_doc
            ],
            doc_ids[
                integer_doc
            ],
        ),
    )

    body_lists = [
        []
        for _ in range(
            term_count
        )
    ]

    prefix_lists = [
        None
        for _ in range(
            term_count
        )
    ]

    new_doc_ids = [
        None
    ] * document_count

    body_tfs_local = body_tfs
    prefix_term_ids_local = prefix_term_ids
    prefix_tfs_local = prefix_tfs
    prefix_offsets_local = prefix_offsets

    for new_integer_doc, old_integer_doc in enumerate(
        order
    ):
        new_doc_ids[
            new_integer_doc
        ] = doc_ids[
            old_integer_doc
        ]

        start = body_offsets_local[
            old_integer_doc
        ]

        end = body_offsets_local[
            old_integer_doc + 1
        ]

        for position in range(
            start,
            end,
        ):
            integer_term = body_term_ids_local[
                position
            ]

            posting = body_lists[
                integer_term
            ]

            posting.append(
                new_integer_doc
            )

            posting.append(
                int(
                    body_tfs_local[
                        position
                    ]
                )
            )

        start = prefix_offsets_local[
            old_integer_doc
        ]

        end = prefix_offsets_local[
            old_integer_doc + 1
        ]

        for position in range(
            start,
            end,
        ):
            integer_term = prefix_term_ids_local[
                position
            ]

            posting = prefix_lists[
                integer_term
            ]

            if posting is None:
                posting = []

                prefix_lists[
                    integer_term
                ] = posting

            posting.append(
                new_integer_doc
            )

            posting.append(
                int(
                    prefix_tfs_local[
                        position
                    ]
                )
            )

    postings = {
        terms[
            integer_term
        ]: body_lists[
            integer_term
        ]
        for integer_term in range(
            term_count
        )
        if body_lists[
            integer_term
        ]
    }

    prefix_postings = {
        terms[
            integer_term
        ]: prefix_lists[
            integer_term
        ]
        for integer_term in range(
            term_count
        )
        if prefix_lists[
            integer_term
        ]
    }

    self.postings = postings

    self.prefix_postings = (
        prefix_postings
    )

    self.doc_len = doc_len

    self.doc_ids = new_doc_ids

    self._integer_postings = True

    self.doc_text = {}

    self.N = document_count

    if document_count:
        self.avg_doc_len = (
            total_doc_len
            / document_count
        )
    else:
        self.avg_doc_len = 0.0


InvertedIndex.build = _v12_build

_V16_PREVIOUS_LOAD = InvertedIndex.load
_V16_BODY = None
_V16_PREFIX = None
_V16_TERMS = None


def _v16_make_ranges(
    body,
    prefix,
    terms,
    workers,
):
    weights = []
    total = 0

    for term in terms:
        weight = (
            len(
                body[
                    term
                ]
            )
            // 2
        )

        prefix_posting = (
            prefix.get(
                term
            )
        )

        if prefix_posting:
            weight += (
                len(
                    prefix_posting
                )
                // 2
            )

        weights.append(
            weight
        )

        total += weight

    ranges = []

    start = 0
    accumulated = 0
    remaining_workers = workers
    remaining_weight = total

    target = (
        remaining_weight
        / remaining_workers
    )

    for integer_term, weight in enumerate(
        weights
    ):
        accumulated += weight

        if (
            remaining_workers > 1
            and accumulated >= target
        ):
            end = integer_term + 1

            ranges.append(
                (
                    start,
                    end,
                )
            )

            remaining_weight -= accumulated
            remaining_workers -= 1

            start = end
            accumulated = 0

            target = (
                remaining_weight
                / remaining_workers
            )

    ranges.append(
        (
            start,
            len(
                terms
            ),
        )
    )

    return ranges


def _v16_encode_range(
    bounds,
):
    start, end = bounds

    body = _V16_BODY
    prefix = _V16_PREFIX
    terms = _V16_TERMS

    body_entries = 0
    prefix_entries = 0

    for integer_term in range(
        start,
        end,
    ):
        term = terms[
            integer_term
        ]

        body_entries += (
            len(
                body[
                    term
                ]
            )
            // 2
        )

        prefix_posting = (
            prefix.get(
                term
            )
        )

        if prefix_posting:
            prefix_entries += (
                len(
                    prefix_posting
                )
                // 2
            )

    body_doc_blob = bytearray()

    body_tf_flags = bytearray(
        (
            body_entries + 7
        )
        // 8
    )

    body_tf_writer = (
        _V7BitWriter()
    )

    body_tf_position = 0
    body_df = []

    for integer_term in range(
        start,
        end,
    ):
        term = terms[
            integer_term
        ]

        posting = body[
            term
        ]

        body_df.append(
            len(
                posting
            )
            // 2
        )

        previous_doc = -1

        for position in range(
            0,
            len(
                posting
            ),
            2,
        ):
            integer_doc = int(
                posting[
                    position
                ]
            )

            tf = int(
                posting[
                    position + 1
                ]
            )

            _write_varint(
                body_doc_blob,
                integer_doc
                - previous_doc,
            )

            previous_doc = (
                integer_doc
            )

            if tf > 1:
                body_tf_flags[
                    body_tf_position
                    >> 3
                ] |= (
                    1
                    << (
                        body_tf_position
                        & 7
                    )
                )

                body_tf_writer.write_gamma(
                    tf - 1
                )

            body_tf_position += 1

    prefix_position_blob = (
        bytearray()
    )

    prefix_tf_flags = bytearray(
        (
            prefix_entries + 7
        )
        // 8
    )

    prefix_tf_writer = (
        _V7BitWriter()
    )

    prefix_tf_position = 0
    prefix_df = []

    for integer_term in range(
        start,
        end,
    ):
        term = terms[
            integer_term
        ]

        prefix_posting = (
            prefix.get(
                term
            )
        )

        if not prefix_posting:
            continue

        body_posting = body[
            term
        ]

        prefix_df.append(
            len(
                prefix_posting
            )
            // 2
        )

        prefix_position = 0

        prefix_length = len(
            prefix_posting
        )

        previous_body_position = -1
        body_position = 0

        for position in range(
            0,
            len(
                body_posting
            ),
            2,
        ):
            body_doc = body_posting[
                position
            ]

            if (
                prefix_position
                < prefix_length
                and prefix_posting[
                    prefix_position
                ] == body_doc
            ):
                _write_varint(
                    prefix_position_blob,
                    body_position
                    - previous_body_position,
                )

                previous_body_position = (
                    body_position
                )

                tf = int(
                    prefix_posting[
                        prefix_position
                        + 1
                    ]
                )

                if tf > 1:
                    prefix_tf_flags[
                        prefix_tf_position
                        >> 3
                    ] |= (
                        1
                        << (
                            prefix_tf_position
                            & 7
                        )
                    )

                    prefix_tf_writer.write_gamma(
                        tf - 1
                    )

                prefix_tf_position += 1
                prefix_position += 2

            body_position += 1

        if prefix_position != (
            prefix_length
        ):
            raise ValueError(
                "Prefix posting not contained in body posting"
            )

    return (
        start,
        end,
        _v7_encode_varints(
            body_df
        ),
        bytes(
            body_doc_blob
        ),
        bytes(
            body_tf_flags
        ),
        body_tf_writer.finish(),
        _v7_encode_varints(
            prefix_df
        ),
        bytes(
            prefix_position_blob
        ),
        bytes(
            prefix_tf_flags
        ),
        prefix_tf_writer.finish(),
    )


def _v16_save(
    self,
    index_dir,
):
    import multiprocessing as mp

    global _V16_BODY
    global _V16_PREFIX
    global _V16_TERMS

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
        and len(
            possible_doc_ids
        ) == self.N
    ):
        doc_ids = list(
            possible_doc_ids
        )
    else:
        doc_ids = list(
            self.doc_len.keys()
        )

    doc_to_int = {
        doc_id: integer_doc
        for integer_doc, doc_id
        in enumerate(
            doc_ids
        )
    }

    if getattr(
        self,
        "_integer_postings",
        False,
    ):
        body_postings = (
            self.postings
        )

        prefix_postings = (
            self.prefix_postings
        )
    else:
        body_postings = (
            _v7_to_integer_postings(
                self.postings,
                doc_to_int,
            )
        )

        prefix_postings = (
            _v7_to_integer_postings(
                self.prefix_postings,
                doc_to_int,
            )
        )

    posting_terms = list(
        body_postings.keys()
    )

    prefix_bitmap = bytearray(
        (
            len(
                posting_terms
            )
            + 7
        )
        // 8
    )

    for integer_term, term in enumerate(
        posting_terms
    ):
        if prefix_postings.get(
            term
        ):
            prefix_bitmap[
                integer_term
                >> 3
            ] |= (
                1
                << (
                    integer_term
                    & 7
                )
            )

    workers = min(
        4,
        os.cpu_count()
        or 1,
    )

    _V16_BODY = (
        body_postings
    )

    _V16_PREFIX = (
        prefix_postings
    )

    _V16_TERMS = (
        posting_terms
    )

    ranges = _v16_make_ranges(
        body_postings,
        prefix_postings,
        posting_terms,
        workers,
    )

    try:
        if workers == 1:
            chunks = [
                _v16_encode_range(
                    ranges[
                        0
                    ]
                )
            ]
        else:
            context = (
                mp.get_context(
                    "fork"
                )
            )

            with context.Pool(
                processes=workers
            ) as pool:
                chunks = pool.map(
                    _v16_encode_range,
                    ranges,
                )
    finally:
        _V16_BODY = None
        _V16_PREFIX = None
        _V16_TERMS = None

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
            self.doc_len[
                doc_id
            ]
            for doc_id
            in doc_ids
        ),
    )

    (
        doc_id_width,
        doc_id_blob,
        fallback_doc_ids,
    ) = _v7_pack_doc_ids(
        doc_ids
    )

    state = {
        "format_version": 16,
        "doc_id_width":
            doc_id_width,
        "doc_id_blob":
            doc_id_blob,
        "doc_lengths":
            doc_lengths,
        "posting_terms":
            posting_terms,
        "prefix_bitmap":
            bytes(
                prefix_bitmap
            ),
        "chunks":
            chunks,
        "N":
            self.N,
        "avg_doc_len":
            self.avg_doc_len,
    }

    if fallback_doc_ids is not None:
        state[
            "doc_ids"
        ] = fallback_doc_ids

    path = os.path.join(
        index_dir,
        _INDEX_FILENAME,
    )

    with bz2.open(
        path,
        "wb",
        compresslevel=9,
    ) as f:
        pickle.dump(
            state,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _v16_load(
    cls,
    index_dir,
):
    path = os.path.join(
        index_dir,
        _INDEX_FILENAME,
    )

    with bz2.open(
        path,
        "rb",
    ) as f:
        state = pickle.load(
            f
        )

    if state.get(
        "format_version",
        2,
    ) < 16:
        return _V16_PREVIOUS_LOAD(
            index_dir
        )

    doc_ids = (
        _v7_unpack_doc_ids(
            state
        )
    )

    doc_lengths = state[
        "doc_lengths"
    ]

    posting_terms = state[
        "posting_terms"
    ]

    prefix_bitmap = state[
        "prefix_bitmap"
    ]

    postings = {}

    for chunk in state[
        "chunks"
    ]:
        (
            start,
            end,
            body_df_blob,
            body_doc_blob,
            body_tf_flags,
            body_tf_blob,
            prefix_df_blob,
            prefix_position_blob,
            prefix_tf_flags,
            prefix_tf_blob,
        ) = chunk

        term_count = (
            end - start
        )

        body_df = (
            _v7_decode_varints(
                body_df_blob,
                term_count,
            )
        )

        body_reader = (
            _V10VarintReader(
                body_doc_blob
            )
        )

        body_tf_reader = (
            _V7BitReader(
                body_tf_blob
            )
        )

        body_tf_position = 0

        for local_term, df in enumerate(
            body_df
        ):
            integer_term = (
                start
                + local_term
            )

            term = posting_terms[
                integer_term
            ]

            posting = {}

            previous_doc = -1

            for _ in range(
                df
            ):
                previous_doc += (
                    body_reader.read()
                )

                tf = 1

                if body_tf_flags[
                    body_tf_position
                    >> 3
                ] & (
                    1
                    << (
                        body_tf_position
                        & 7
                    )
                ):
                    tf = (
                        body_tf_reader.read_gamma()
                        + 1
                    )

                posting[
                    doc_ids[
                        previous_doc
                    ]
                ] = tf

                body_tf_position += 1

            postings[
                term
            ] = posting

        if body_reader.position != len(
            body_doc_blob
        ):
            raise ValueError(
                "Invalid V16 body stream"
            )

    prefix_postings = {}

    for chunk in state[
        "chunks"
    ]:
        (
            start,
            end,
            body_df_blob,
            body_doc_blob,
            body_tf_flags,
            body_tf_blob,
            prefix_df_blob,
            prefix_position_blob,
            prefix_tf_flags,
            prefix_tf_blob,
        ) = chunk

        prefix_term_count = 0

        for integer_term in range(
            start,
            end,
        ):
            if prefix_bitmap[
                integer_term >> 3
            ] & (
                1
                << (
                    integer_term
                    & 7
                )
            ):
                prefix_term_count += 1

        prefix_df = (
            _v7_decode_varints(
                prefix_df_blob,
                prefix_term_count,
            )
        )

        prefix_reader = (
            _V10VarintReader(
                prefix_position_blob
            )
        )

        prefix_tf_reader = (
            _V7BitReader(
                prefix_tf_blob
            )
        )

        prefix_metadata_position = 0
        prefix_tf_position = 0

        for integer_term in range(
            start,
            end,
        ):
            if not (
                prefix_bitmap[
                    integer_term
                    >> 3
                ]
                & (
                    1
                    << (
                        integer_term
                        & 7
                    )
                )
            ):
                continue

            term = posting_terms[
                integer_term
            ]

            df = prefix_df[
                prefix_metadata_position
            ]

            body_doc_ids = list(
                postings[
                    term
                ].keys()
            )

            posting = {}

            previous_position = -1

            for _ in range(
                df
            ):
                previous_position += (
                    prefix_reader.read()
                )

                tf = 1

                if prefix_tf_flags[
                    prefix_tf_position
                    >> 3
                ] & (
                    1
                    << (
                        prefix_tf_position
                        & 7
                    )
                ):
                    tf = (
                        prefix_tf_reader.read_gamma()
                        + 1
                    )

                posting[
                    body_doc_ids[
                        previous_position
                    ]
                ] = tf

                prefix_tf_position += 1

            prefix_postings[
                term
            ] = posting

            prefix_metadata_position += 1

        if prefix_reader.position != len(
            prefix_position_blob
        ):
            raise ValueError(
                "Invalid V16 prefix stream"
            )

        if prefix_metadata_position != (
            prefix_term_count
        ):
            raise ValueError(
                "Invalid V16 prefix metadata"
            )

    index = cls()

    index.postings = postings

    index.prefix_postings = (
        prefix_postings
    )

    index.doc_len = {
        doc_id: int(
            length
        )
        for doc_id, length
        in zip(
            doc_ids,
            doc_lengths,
        )
    }

    index.doc_ids = doc_ids

    index._integer_postings = False

    index.N = int(
        state[
            "N"
        ]
    )

    index.avg_doc_len = float(
        state[
            "avg_doc_len"
        ]
    )

    index.doc_text = {}

    return index


InvertedIndex.save = _v16_save

InvertedIndex.load = classmethod(
    _v16_load
)
