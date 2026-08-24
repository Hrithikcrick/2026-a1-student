import csv
import time

from harness.metrics import evaluate_run
from harness.trec_io import read_queries, read_qrels
from submission import bm25
from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex

CORPUS_PATH = "data/full/corpus.jsonl"
QUERIES_PATH = "data/full/queries_dev.tsv"
QRELS_PATH = "data/full/qrels_dev.txt"
OUTPUT_PATH = "runs/bm25_extended_tuning.csv"

K1_VALUES = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
B_VALUES = [0.50, 0.55, 0.60, 0.65]


def save_results(results):
    results.sort(
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
    print("Loading corpus...")
    corpus = load_corpus(CORPUS_PATH)

    print("Building index...")
    index = InvertedIndex()
    index.build(corpus)

    index.doc_text = {}
    del corpus

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
                    b=b,
                )

            report = evaluate_run(run, qrels, k=10)
            metrics = report["aggregate"]

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
                "seconds": time.perf_counter() - start,
            }

            results.append(row)
            save_results(results)

            print(
                f"[{current}/{total}] "
                f"k1={k1:.1f} "
                f"b={b:.2f} "
                f"nDCG={row['ndcg@10']:.6f} "
                f"MAP={row['map@10']:.6f}"
            )

    results.sort(
        key=lambda x: (-x["ndcg@10"], -x["map@10"])
    )

    print()
    print("TOP 10")

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