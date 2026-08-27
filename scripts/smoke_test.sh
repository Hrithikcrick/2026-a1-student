#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=python

echo "== Interface conformance tests =="

$PYTHON -m pytest tests/test_interface_conformance.py -v

echo

echo "== Metrics unit tests =="

$PYTHON -m pytest tests/test_metrics.py -v

echo

echo "== Full harness run on the toy set =="

$PYTHON -m harness.run_harness \
  --corpus data/toy/corpus.jsonl \
  --queries data/toy/queries_dev.tsv \
  --qrels data/toy/qrels_dev.txt \
  --baseline-run data/toy/reference_bm25_run_dev.trec \
  --run-out runs/dev_run.trec \
  --report-out runs/dev_report.json

echo

echo "All smoke checks passed."