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
OUTPUT_PATH = "runs/bm25_tuning.csv"

K1_VALUES = [0.6, 0.9, 1.2, 1.5, 1.8, 2.1]
B_VALUES = [0.2, 0.4, 0.6, 0.75, 0.9]


def save_results(results):
    ordered = sorted(
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
        writer.writerows(ordered)


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
    experiment = 0

    for k1 in K1_VALUES:
        for b in B_VALUES:
            experiment += 1

            print(
                f"[{experiment}/{total}] "
                f"k1={k1:.2f} b={b:.2f}"
            )

            start = time.perf_counter()

            run = {}

            for qid, query in queries:
                run[qid] = bm25.score(
                    query,
                    10,
                    k1=k1,
                    b=b,
                )

            report = evaluate_run(
                run,
                qrels,
                k=10,
            )

            metrics = report["aggregate"]

            ndcg = metrics["ndcg@10"]
            map10 = metrics["map@10"]
            mrr = metrics["mrr"]
            p10 = metrics["p@10"]

            elapsed = time.perf_counter() - start

            weighted = (
                0.7 * ndcg
                + 0.1 * map10
            )

            row = {
                "k1": k1,
                "b": b,
                "ndcg@10": ndcg,
                "map@10": map10,
                "mrr": mrr,
                "p@10": p10,
                "weighted_score": weighted,
                "seconds": elapsed,
            }

            results.append(row)

            print(
                f"nDCG@10={ndcg:.6f} "
                f"MAP@10={map10:.6f} "
                f"MRR={mrr:.6f} "
                f"P@10={p10:.6f} "
                f"time={elapsed:.2f}s"
            )

            save_results(results)

    results.sort(
        key=lambda x: (
            -x["ndcg@10"],
            -x["map@10"],
        )
    )

    print()
    print("TOP 10")
    print()

    for i, row in enumerate(results[:10], start=1):
        print(
            f"{i:2d}. "
            f"k1={row['k1']:.2f} "
            f"b={row['b']:.2f} "
            f"nDCG@10={row['ndcg@10']:.6f} "
            f"MAP@10={row['map@10']:.6f} "
            f"P@10={row['p@10']:.6f}"
        )

    print()
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()