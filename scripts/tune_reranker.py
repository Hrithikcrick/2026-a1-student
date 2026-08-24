import csv
import math
import os
import time

from harness.metrics import evaluate_run
from harness.trec_io import read_queries, read_qrels
from submission import bm25, boolean_vsm
from submission.corpus_utils import load_corpus
from submission.indexer import InvertedIndex, tokenize


CORPUS_PATH = "data/full/corpus.jsonl"
QUERIES_PATH = "data/full/queries_dev.tsv"
QRELS_PATH = "data/full/qrels_dev.txt"

INDEX_DIR = "runs/tuning_index"
OUTPUT_PATH = "runs/reranker_tuning.csv"

K1 = 2.4
B = 0.60
CANDIDATE_K = 100

VSM_WEIGHTS = [0.0, 0.05, 0.10, 0.20, 0.30]
COVERAGE_WEIGHTS = [0.0, 0.05, 0.10, 0.20, 0.30]
RARE_WEIGHTS = [0.0, 0.05, 0.10, 0.20]


def get_index():
    path = os.path.join(INDEX_DIR, "inverted_index.pkl")

    if os.path.exists(path):
        print("Loading cached tuning index...")
        return InvertedIndex.load(INDEX_DIR)

    print("Building tuning index...")
    corpus = load_corpus(CORPUS_PATH)

    start = time.perf_counter()

    index = InvertedIndex()
    index.build(corpus)
    index.save(INDEX_DIR)

    print(f"Index built in {time.perf_counter() - start:.2f}s")

    return index


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

            if query_terms:
                coverage = len(matched) / len(query_terms)
            else:
                coverage = 0.0

            if total_idf > 0:
                rare_coverage = (
                    sum(idf_values[t] for t in matched)
                    / total_idf
                )
            else:
                rare_coverage = 0.0

            features[doc_id] = {
                "bm25": bm25_scores.get(doc_id, 0.0) / max_bm25,
                "vsm": vsm_scores.get(doc_id, 0.0),
                "coverage": coverage,
                "rare": rare_coverage,
            }

        all_features[qid] = features

    return all_features


def build_run(all_features, vsm_weight, coverage_weight, rare_weight):
    run = {}

    for qid, docs in all_features.items():
        ranked = []

        for doc_id, feature in docs.items():
            score = (
                feature["bm25"]
                + vsm_weight * feature["vsm"]
                + coverage_weight * feature["coverage"]
                + rare_weight * feature["rare"]
            )

            ranked.append((doc_id, score))

        ranked.sort(key=lambda x: (-x[1], x[0]))

        run[qid] = ranked[:10]

    return run


def save_results(results):
    results.sort(
        key=lambda x: (-x["ndcg@10"], -x["map@10"])
    )

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "vsm_weight",
                "coverage_weight",
                "rare_weight",
                "ndcg@10",
                "map@10",
                "mrr",
                "p@10",
            ],
        )

        writer.writeheader()
        writer.writerows(results)


def main():
    index = get_index()

    queries = read_queries(QUERIES_PATH)
    qrels = read_qrels(QRELS_PATH)

    print("Building query features...")
    features = make_features(index, queries)

    results = []

    total = (
        len(VSM_WEIGHTS)
        * len(COVERAGE_WEIGHTS)
        * len(RARE_WEIGHTS)
    )

    current = 0

    for vsm_weight in VSM_WEIGHTS:
        for coverage_weight in COVERAGE_WEIGHTS:
            for rare_weight in RARE_WEIGHTS:
                current += 1

                run = build_run(
                    features,
                    vsm_weight,
                    coverage_weight,
                    rare_weight
                )

                report = evaluate_run(
                    run,
                    qrels,
                    k=10
                )

                metrics = report["aggregate"]

                row = {
                    "vsm_weight": vsm_weight,
                    "coverage_weight": coverage_weight,
                    "rare_weight": rare_weight,
                    "ndcg@10": metrics["ndcg@10"],
                    "map@10": metrics["map@10"],
                    "mrr": metrics["mrr"],
                    "p@10": metrics["p@10"],
                }

                results.append(row)

                print(
                    f"[{current}/{total}] "
                    f"vsm={vsm_weight:.2f} "
                    f"coverage={coverage_weight:.2f} "
                    f"rare={rare_weight:.2f} "
                    f"nDCG={row['ndcg@10']:.6f}"
                )

    save_results(results)

    results.sort(
        key=lambda x: (-x["ndcg@10"], -x["map@10"])
    )

    print()
    print("TOP 15")
    print()

    for i, row in enumerate(results[:15], 1):
        print(
            f"{i}. "
            f"vsm={row['vsm_weight']} "
            f"coverage={row['coverage_weight']} "
            f"rare={row['rare_weight']} "
            f"nDCG@10={row['ndcg@10']:.6f} "
            f"MAP@10={row['map@10']:.6f} "
            f"MRR={row['mrr']:.6f} "
            f"P@10={row['p@10']:.6f}"
        )


if __name__ == "__main__":
    main()