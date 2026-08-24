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
# #1035: generation-loop metrics. Optional — a matmul microbenchmark has no decode loop — but a
# record that claims to have run one must carry the numbers a dashboard plots, with plausible
# values. A silently empty or negative field is the failure mode this catches.
required_generation = {"prefill_tokens","decode_steps","ms_per_decode_step","bytes_read","adapter_count","adapter_bytes"}
non_negative_generation = required_generation | {"decode_tokens_per_second","prefill_tokens_per_second",
                                                "ttft_ms","effective_bandwidth_bytes_per_second",
                                                "bandwidth_utilization_percent","kernel_share_of_decode_percent",
                                                "page_faults","page_faults_per_second"}
fail = 0

def check_generation(path, gen):
    """Returns a list of problems with a record's `generation` block."""
    problems = []
    missing = required_generation - gen.keys()
    if missing:
        problems.append(f"generation missing {sorted(missing)}")
        return problems
    for key in sorted(non_negative_generation & gen.keys()):
        value = gen[key]
        if value is None:
            continue
        if not isinstance(value, (int, float)) or value < 0:
            problems.append(f"generation.{key}={value!r} must be a non-negative number or null")
    for key in ("bandwidth_utilization_percent", "kernel_share_of_decode_percent"):
        value = gen.get(key)
        if isinstance(value, (int, float)) and value > 100.0:
            problems.append(f"generation.{key}={value} exceeds 100%")
    if gen["decode_steps"] > 0 and gen.get("decode_tokens_per_second") is None and gen["ms_per_decode_step"] == 0:
        problems.append("generation: decode steps were recorded but never timed")
    breakdown = gen.get("module_breakdown_ms", {})
    if not isinstance(breakdown, dict):
        problems.append("generation.module_breakdown_ms must be an object of module -> milliseconds")
    else:
        for module, ms in breakdown.items():
            if not isinstance(ms, (int, float)) or ms < 0:
                problems.append(f"generation.module_breakdown_ms[{module!r}]={ms!r} must be non-negative")
    return problems
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
    gen = rec.get("generation")
    if gen is not None:
        problems = check_generation(path, gen)
        if problems:
            for p in problems:
                print(f"FAIL {path}: {p}", file=sys.stderr)
            fail += 1
            continue
    suffix = ""
    if gen is not None:
        tok = gen.get("decode_tokens_per_second")
        bw = gen.get("effective_bandwidth_bytes_per_second")
        suffix = "  decode=" + (f"{tok:.2f} tok/s" if isinstance(tok, (int, float)) else "n/a")
        suffix += "  bw=" + (f"{bw / 1e9:.2f} GB/s" if isinstance(bw, (int, float)) else "n/a")
    print(f"OK   {path}  {rec['scenario']}  mean={rec['metrics']['value_mean']:.4f} {rec['metrics']['unit']}{suffix}")
sys.exit(1 if fail else 0)
PY
