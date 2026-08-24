import csv

from harness.metrics import evaluate_run
from harness.trec_io import read_queries, read_qrels, read_run

QUERIES_PATH = "data/full/queries_dev.tsv"
QRELS_PATH = "data/full/qrels_dev.txt"
RUN_PATH = "runs/best_bm25_run.trec"
OUTPUT_PATH = "runs/error_analysis.csv"


def main():
    queries = dict(read_queries(QUERIES_PATH))
    qrels = read_qrels(QRELS_PATH)
    run = read_run(RUN_PATH)

    report = evaluate_run(run, qrels, k=10)
    per_query = report["per_query"]

    rows = []

    for qid, metrics in per_query.items():
        ranked = run.get(qid, [])[:10]
        judgments = qrels.get(qid, {})

        rels = [
            judgments.get(doc_id, 0)
            for doc_id, _ in ranked
        ]

        rows.append({
            "qid": qid,
            "query": queries.get(qid, ""),
            "ndcg@10": metrics["ndcg@10"],
            "map@10": metrics["map@10"],
            "mrr": metrics["mrr"],
            "p@10": metrics["p@10"],
            "top10_relevance": " ".join(map(str, rels)),
        })

    rows.sort(key=lambda x: x["ndcg@10"])

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "qid",
                "query",
                "ndcg@10",
                "map@10",
                "mrr",
                "p@10",
                "top10_relevance",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("WORST 15 QUERIES")
    print()

    for i, row in enumerate(rows[:15], 1):
        print(f"{i}. {row['qid']}")
        print(f"Query: {row['query']}")
        print(f"nDCG@10: {row['ndcg@10']:.6f}")
        print(f"MAP@10: {row['map@10']:.6f}")
        print(f"P@10: {row['p@10']:.6f}")
        print(f"Top10 relevance: {row['top10_relevance']}")
        print()


if __name__ == "__main__":
    main()