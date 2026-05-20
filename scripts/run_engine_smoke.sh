#!/usr/bin/env bash
# Convenience wrapper: smoke-mode engine benchmarks for CI on ubuntu-latest.
# Same as `run_engine_benchmarks.sh --smoke` but with a stable output path
# that CI workflows can upload as an artifact without guessing a timestamp.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$REPO_ROOT/scripts/run_engine_benchmarks.sh" --smoke --out-dir "$REPO_ROOT/out/engine/smoke" "$@"
