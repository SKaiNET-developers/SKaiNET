#!/usr/bin/env bash
# Asserts that every BenchmarkRecord JSON under the given directory carries
# the expected top-level fields. Catches schema regressions in CI before
# downstream tooling (PTS uploads, dashboards) silently drops fields.
set -euo pipefail

DIR="${1:-out/engine/smoke}"
EXPECTED_SCHEMA="${EXPECTED_SCHEMA:-1.0.0}"

if [ ! -d "$DIR" ]; then
    echo "ERROR: directory not found: $DIR" >&2
    exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 required for JSON schema check" >&2
    exit 2
fi

shopt -s nullglob
FILES=("$DIR"/*.json)
if [ ${#FILES[@]} -eq 0 ]; then
    echo "ERROR: no JSON files under $DIR" >&2
    exit 1
fi

python3 - "$EXPECTED_SCHEMA" "${FILES[@]}" <<'PY'
import json, sys
expected_schema = sys.argv[1]
required_top = {"schema_version","suite","scenario","published_at","runtime","system","config","metrics","samples"}
required_metrics = {"primary_metric","unit","value_mean","value_stddev","value_min","value_max","cov_percent"}
required_runtime = {"name","version","commit","backend","kernel_provider","available_providers"}
fail = 0
for path in sys.argv[2:]:
    try:
        with open(path) as f:
            rec = json.load(f)
    except Exception as e:
        print(f"FAIL {path}: not valid JSON ({e})", file=sys.stderr); fail += 1; continue
    missing = required_top - rec.keys()
    if missing:
        print(f"FAIL {path}: missing top-level fields {sorted(missing)}", file=sys.stderr); fail += 1; continue
    if rec["schema_version"] != expected_schema:
        print(f"FAIL {path}: schema_version={rec['schema_version']!r} != expected {expected_schema!r}", file=sys.stderr); fail += 1; continue
    mm = required_metrics - rec["metrics"].keys()
    if mm:
        print(f"FAIL {path}: metrics missing {sorted(mm)}", file=sys.stderr); fail += 1; continue
    rm = required_runtime - rec["runtime"].keys()
    if rm:
        print(f"FAIL {path}: runtime missing {sorted(rm)}", file=sys.stderr); fail += 1; continue
    print(f"OK   {path}  {rec['scenario']}  mean={rec['metrics']['value_mean']:.4f} {rec['metrics']['unit']}")
sys.exit(1 if fail else 0)
PY
