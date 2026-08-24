import csv
import math
import os

from harness.metrics import evaluate_run
from harness.trec_io import read_queries, read_qrels
from submission import bm25, boolean_vsm
from submission.indexer import InvertedIndex, tokenize


QUERIES_PATH = "data/full/queries_dev.tsv"
QRELS_PATH = "data/full/qrels_dev.txt"
INDEX_DIR = "runs/tuning_index"
OUTPUT_PATH = "runs/reranker_fine_tuning.csv"

K1 = 2.4
B = 0.60
CANDIDATE_K = 100

VSM_WEIGHTS = [0.15, 0.175, 0.20, 0.225, 0.25]
COVERAGE_WEIGHTS = [0.15, 0.175, 0.20, 0.225, 0.25]
RARE_WEIGHTS = [0.0, 0.025, 0.05, 0.075]


def make_features(index, queries):
    bm25.build(index)
    boolean_vsm.build(index)

    all_features = {}

    for number, (qid, query) in enumerate(queries, 1):
        print(f"Features {number}/{len(queries)}")

        bm25_results = bm25.score(
            query,
            CANDIDATE_K,
            k1=K1,
            b=B
        )

        vsm_results = boolean_vsm.vsm_score(
            query,
            CANDIDATE_K
        )

        bm25_scores = dict(bm25_results)
        vsm_scores = dict(vsm_results)

        candidates = set(bm25_scores)
        candidates.update(vsm_scores)

        max_bm25 = max(bm25_scores.values(), default=1.0)

        if max_bm25 <= 0:
            max_bm25 = 1.0

        query_terms = list(dict.fromkeys(tokenize(query)))

        idf_values = {}

        for term in query_terms:
            df = index.document_frequency(term)

            if df > 0:
                idf_values[term] = math.log(
                    ((index.N - df + 0.5) / (df + 0.5)) + 1.0
                )
            else:
                idf_values[term] = 0.0

        total_idf = sum(idf_values.values())

        features = {}

        for doc_id in candidates:
            matched = []

            for term in query_terms:
                posting = index.postings.get(term)

                if posting and doc_id in posting:
                    matched.append(term)

            coverage = (
                len(matched) / len(query_terms)
                if query_terms
                else 0.0
            )

            rare = (
                sum(idf_values[t] for t in matched) / total_idf
                if total_idf > 0
                else 0.0
            )

            features[doc_id] = (
                bm25_scores.get(doc_id, 0.0) / max_bm25,
                vsm_scores.get(doc_id, 0.0),
                coverage,
                rare
            )

        all_features[qid] = features

    return all_features


def build_run(features, vw, cw, rw):
    run = {}

    for qid, docs in features.items():
        results = []

        for doc_id, values in docs.items():
            bscore, vscore, coverage, rare = values

            score = (
                bscore
                + vw * vscore
                + cw * coverage
                + rw * rare
            )

            results.append((doc_id, score))

        results.sort(key=lambda x: (-x[1], x[0]))
        run[qid] = results[:10]

    return run


def main():
    index = InvertedIndex.load(INDEX_DIR)

    queries = read_queries(QUERIES_PATH)
    qrels = read_qrels(QRELS_PATH)

    print("Building features...")
    features = make_features(index, queries)

    rows = []

    total = (
        len(VSM_WEIGHTS)
        * len(COVERAGE_WEIGHTS)
        * len(RARE_WEIGHTS)
    )

    current = 0

    for vw in VSM_WEIGHTS:
        for cw in COVERAGE_WEIGHTS:
            for rw in RARE_WEIGHTS:
                current += 1

                run = build_run(
                    features,
                    vw,
                    cw,
                    rw
                )

                report = evaluate_run(
                    run,
                    qrels,
                    k=10
                )

                metrics = report["aggregate"]

                row = {
                    "vsm": vw,
                    "coverage": cw,
                    "rare": rw,
                    "ndcg@10": metrics["ndcg@10"],
                    "map@10": metrics["map@10"],
                    "mrr": metrics["mrr"],
                    "p@10": metrics["p@10"],
                }

                rows.append(row)

                print(
                    f"[{current}/{total}] "
                    f"vsm={vw:.3f} "
                    f"coverage={cw:.3f} "
                    f"rare={rw:.3f} "
                    f"nDCG={row['ndcg@10']:.6f}"
                )

    rows.sort(
        key=lambda x: (-x["ndcg@10"], -x["map@10"])
    )

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "vsm",
                "coverage",
                "rare",
                "ndcg@10",
                "map@10",
                "mrr",
                "p@10",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("TOP 15")
    print()

    for i, row in enumerate(rows[:15], 1):
        print(
            f"{i}. "
            f"vsm={row['vsm']} "
            f"coverage={row['coverage']} "
            f"rare={row['rare']} "
            f"nDCG@10={row['ndcg@10']:.6f} "
            f"MAP@10={row['map@10']:.6f} "
            f"MRR={row['mrr']:.6f} "
            f"P@10={row['p@10']:.6f}"
        )


if __name__ == "__main__":
    main()