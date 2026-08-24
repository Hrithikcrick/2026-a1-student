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
OUTPUT_PATH = "runs/porter_stopwords_bm25_tuning.csv"

K1_VALUES = [2.4, 2.8, 3.2, 3.4, 3.6, 4.0, 4.4]
B_VALUES = [0.40, 0.50, 0.60, 0.70]

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


def tokenizer(text):
    tokens = TOKEN_RE.findall(text.lower())
    return [
        STEMMER.stem(token)
        for token in tokens
        if token not in STOPWORDS
    ]


def save_results(results):
    results = sorted(
        results,
        key=lambda x: (-x["ndcg@10"], -x["map@10"])
    )

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "k1",
                "b",
                "ndcg@10",
                "map@10",
                "mrr",
                "p@10",
                "weighted_score",
                "seconds",
            ],
        )

        writer.writeheader()
        writer.writerows(results)


def main():
    indexer_module.tokenize = tokenizer
    bm25.tokenize = tokenizer

    print("Loading corpus...")
    corpus = load_corpus(CORPUS_PATH)

    print("Building Porter + stopword index...")
    start = time.perf_counter()

    index = InvertedIndex()
    index.build(corpus)

    build_time = time.perf_counter() - start

    print(f"Index built in {build_time:.2f}s")
    print(f"Documents: {index.N}")
    print(f"Average length: {index.avg_doc_len:.2f}")

    index.doc_text = {}
    del corpus
    gc.collect()

    bm25.build(index)

    queries = read_queries(QUERIES_PATH)
    qrels = read_qrels(QRELS_PATH)

    results = []

    total = len(K1_VALUES) * len(B_VALUES)
    current = 0

    for k1 in K1_VALUES:
        for b in B_VALUES:
            current += 1

            start = time.perf_counter()
            run = {}

            for qid, query in queries:
                run[qid] = bm25.score(
                    query,
                    10,
                    k1=k1,
                    b=b
                )

            report = evaluate_run(run, qrels, k=10)
            metrics = report["aggregate"]

            elapsed = time.perf_counter() - start

            row = {
                "k1": k1,
                "b": b,
                "ndcg@10": metrics["ndcg@10"],
                "map@10": metrics["map@10"],
                "mrr": metrics["mrr"],
                "p@10": metrics["p@10"],
                "weighted_score":
                    0.7 * metrics["ndcg@10"]
                    + 0.1 * metrics["map@10"],
                "seconds": elapsed,
            }

            results.append(row)
            save_results(results)

            print(
                f"[{current}/{total}] "
                f"k1={k1:.1f} "
                f"b={b:.2f} "
                f"nDCG={row['ndcg@10']:.6f} "
                f"MAP={row['map@10']:.6f} "
                f"P10={row['p@10']:.6f}"
            )

    results.sort(
        key=lambda x: (-x["ndcg@10"], -x["map@10"])
    )

    print()
    print("TOP 10")
    print()

    for i, row in enumerate(results[:10], 1):
        print(
            f"{i}. "
            f"k1={row['k1']} "
            f"b={row['b']} "
            f"nDCG@10={row['ndcg@10']:.6f} "
            f"MAP@10={row['map@10']:.6f} "
            f"MRR={row['mrr']:.6f} "
            f"P@10={row['p@10']:.6f}"
        )


if __name__ == "__main__":
    main()