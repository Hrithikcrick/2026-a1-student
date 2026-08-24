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
OUTPUT_PATH = "runs/bm25_fine_tuning.csv"

K1_VALUES = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
B_VALUES = [0.45, 0.50, 0.55, 0.60, 0.65]


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
    print("Loading corpus...")
    corpus = load_corpus(CORPUS_PATH)

    print("Building index...")
    start = time.perf_counter()

    index = InvertedIndex()
    index.build(corpus)

    print(
        f"Index built in {time.perf_counter() - start:.2f}s "
        f"with {index.N} documents"
    )

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

            print(f"[{current}/{total}] k1={k1} b={b}")

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

            ndcg = metrics["ndcg@10"]
            map10 = metrics["map@10"]
            mrr = metrics["mrr"]
            p10 = metrics["p@10"]

            elapsed = time.perf_counter() - start

            row = {
                "k1": k1,
                "b": b,
                "ndcg@10": ndcg,
                "map@10": map10,
                "mrr": mrr,
                "p@10": p10,
                "weighted_score": 0.7 * ndcg + 0.1 * map10,
                "seconds": elapsed,
            }

            results.append(row)
            save_results(results)

            print(
                f"nDCG@10={ndcg:.6f} "
                f"MAP@10={map10:.6f} "
                f"MRR={mrr:.6f} "
                f"P@10={p10:.6f}"
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