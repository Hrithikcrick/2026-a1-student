import csv
import gc
import re
import time

from nltk.stem import PorterStemmer

from harness.metrics import evaluate_run
from harness.trec_io import read_queries, read_qrels
from submission import bm25
import submission.indexer as indexer_module
from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex


CORPUS_PATH = "data/full/corpus.jsonl"
QUERIES_PATH = "data/full/queries_dev.tsv"
QRELS_PATH = "data/full/qrels_dev.txt"
OUTPUT_PATH = "runs/preprocessing_tuning.csv"

K1 = 3.4
B = 0.60

TOKEN_RE = re.compile(r"[a-z0-9]+")
STEMMER = PorterStemmer()

STOPWORDS = {
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
    "which", "while", "who", "why", "will", "with", "would", "you",
    "your", "yours"
}


def raw_tokenize(text):
    return TOKEN_RE.findall(text.lower())


def stop_tokenize(text):
    tokens = TOKEN_RE.findall(text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def stem_tokenize(text):
    tokens = TOKEN_RE.findall(text.lower())
    return [STEMMER.stem(token) for token in tokens]


def stem_stop_tokenize(text):
    tokens = TOKEN_RE.findall(text.lower())
    return [
        STEMMER.stem(token)
        for token in tokens
        if token not in STOPWORDS
    ]


TOKENIZERS = [
    ("raw", raw_tokenize),
    ("stopwords", stop_tokenize),
    ("porter", stem_tokenize),
    ("porter_stopwords", stem_stop_tokenize),
]


def save_results(results):
    ordered = sorted(
        results,
        key=lambda x: (-x["ndcg@10"], -x["map@10"])
    )

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "preprocessing",
                "k1",
                "b",
                "ndcg@10",
                "map@10",
                "mrr",
                "p@10",
                "build_seconds",
                "query_seconds",
            ],
        )

        writer.writeheader()
        writer.writerows(ordered)


def main():
    print("Loading corpus...")
    corpus = load_corpus(CORPUS_PATH)

    queries = read_queries(QUERIES_PATH)
    qrels = read_qrels(QRELS_PATH)

    results = []

    for number, (name, tokenizer) in enumerate(TOKENIZERS, 1):
        print()
        print(f"[{number}/{len(TOKENIZERS)}] {name}")

        indexer_module.tokenize = tokenizer
        bm25.tokenize = tokenizer

        start = time.perf_counter()

        index = InvertedIndex()
        index.build(corpus)
        index.doc_text = {}

        build_seconds = time.perf_counter() - start

        bm25.build(index)

        start = time.perf_counter()

        run = {}

        for qid, query in queries:
            run[qid] = bm25.score(
                query,
                10,
                k1=K1,
                b=B,
            )

        query_seconds = time.perf_counter() - start

        report = evaluate_run(run, qrels, k=10)
        metrics = report["aggregate"]

        row = {
            "preprocessing": name,
            "k1": K1,
            "b": B,
            "ndcg@10": metrics["ndcg@10"],
            "map@10": metrics["map@10"],
            "mrr": metrics["mrr"],
            "p@10": metrics["p@10"],
            "build_seconds": build_seconds,
            "query_seconds": query_seconds,
        }

        results.append(row)
        save_results(results)

        print(
            f"nDCG@10={row['ndcg@10']:.6f} "
            f"MAP@10={row['map@10']:.6f} "
            f"MRR={row['mrr']:.6f} "
            f"P@10={row['p@10']:.6f}"
        )

        bm25._INDEX = None
        bm25._IDF = {}

        del index
        gc.collect()

    results.sort(
        key=lambda x: (-x["ndcg@10"], -x["map@10"])
    )

    print()
    print("RESULTS")
    print()

    for i, row in enumerate(results, 1):
        print(
            f"{i}. "
            f"{row['preprocessing']} "
            f"nDCG@10={row['ndcg@10']:.6f} "
            f"MAP@10={row['map@10']:.6f} "
            f"MRR={row['mrr']:.6f} "
            f"P@10={row['p@10']:.6f}"
        )


if __name__ == "__main__":
    main()