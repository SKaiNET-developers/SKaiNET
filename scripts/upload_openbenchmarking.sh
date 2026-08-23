#!/usr/bin/env bash
# Publish a full-mode engine benchmark run to OpenBenchmarking.org via Phoronix Test Suite.
#
# One-time setup per bench host (not handled by this script — see
# docs/modules/ROOT/pages/contributing/benchmarks.adoc):
#   1. ./scripts/install_pts.sh
#   2. phoronix-test-suite openbenchmarking-login
#      (interactive; stores the account session locally — this script never sees or
#      handles OpenBenchmarking.org credentials)
#
# What this script does:
#   1. Validates a full-mode JSON run (from scripts/run_engine_benchmarks.sh) against the
#      BenchmarkRecord schema and refuses to proceed if any record is "unstable": true
#      (CoV > benchmarks/manifests/engine-release.yml's stability.cov_limit_percent).
#      Re-run on a quiet, dedicated host if this gate fails.
#   2. Syncs + statically validates the local PTS profiles/suite (scripts/validate_pts_profiles.sh).
#   3. Re-runs the suite through PTS itself (`phoronix-test-suite batch-benchmark`) — PTS needs
#      to produce its own native result to upload; the JSON from step 1 is a parallel, richer
#      record kept for schema validation and historical diffing, not a PTS result file. Batch
#      mode is configured here (non-interactively, upload OFF) rather than depending on a
#      prior `phoronix-test-suite batch-setup` on the host.
#   4. Uploads the resulting PTS result to OpenBenchmarking.org — a public, essentially
#      irreversible action, so it only runs with --confirm (or CONFIRM_PUBLISH=yes); without
#      it, this prints what it would do and exits.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
    sed -n '2,25p' "$0"
    echo
    echo "Usage: $0 <path to a full-mode out/engine/<TIMESTAMP> dir> [--confirm]"
}

JSON_DIR="${1:-}"
if [ -z "$JSON_DIR" ] || [ "$JSON_DIR" = "-h" ] || [ "$JSON_DIR" = "--help" ]; then
    usage
    exit "$([ -z "$JSON_DIR" ] && echo 2 || echo 0)"
fi

CONFIRM="${CONFIRM_PUBLISH:-no}"
[ "${2:-}" = "--confirm" ] && CONFIRM="yes"

RESULT_IDENTIFIER="${RESULT_IDENTIFIER:-skainet-engine-$(basename "$JSON_DIR")}"

if ! command -v phoronix-test-suite >/dev/null 2>&1; then
    echo "ERROR: phoronix-test-suite not installed. Run ./scripts/install_pts.sh first." >&2
    exit 2
fi

echo "== Stage 1: validate the JSON harness run at $JSON_DIR =="
"$REPO_ROOT/scripts/check_engine_json.sh" "$JSON_DIR"

UNSTABLE="$(python3 - "$JSON_DIR" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
bad = [p for p in sorted(glob.glob(os.path.join(d, "*.json"))) if json.load(open(p)).get("unstable")]
print("\n".join(bad))
PY
)"
if [ -n "$UNSTABLE" ]; then
    echo "ERROR: unstable records present (CoV over the manifest's stability limit) —" >&2
    echo "       re-run on a quiet, dedicated bench host before publishing:" >&2
    echo "$UNSTABLE" >&2
    exit 1
fi
echo "Stability gate: OK — no unstable records."

echo
echo "== Stage 2: sync + validate local PTS profiles and suite =="
"$REPO_ROOT/scripts/validate_pts_profiles.sh"

echo
echo "== Stage 3: configure PTS batch mode (non-interactive, upload OFF for this stage) =="
python3 - <<'PY'
import os
import xml.etree.ElementTree as ET

path = os.path.expanduser("~/.phoronix-test-suite/user-config.xml")
os.makedirs(os.path.dirname(path), exist_ok=True)

if os.path.isfile(path):
    tree = ET.parse(path)
    root = tree.getroot()
else:
    root = ET.Element("PhoronixTestSuite")
    tree = ET.ElementTree(root)

def ensure_path(parent, *tags):
    for tag in tags:
        child = parent.find(tag)
        if child is None:
            child = ET.SubElement(parent, tag)
        parent = child
    return parent

batch = ensure_path(root, "Options", "BatchMode")
values = {
    "Configured": "TRUE",
    "SaveResults": "TRUE",
    "OpenBrowser": "FALSE",
    "UploadResults": "FALSE",  # upload is Stage 4, explicit and gated — never a side effect here
    "PromptForTestIdentifier": "FALSE",
    "PromptForTestDescription": "FALSE",
    "PromptSaveName": "FALSE",
    "RunAllTestCombinations": "TRUE",  # runs both panama/scalar for the 3 provider-optioned profiles
}
for tag, value in values.items():
    ensure_path(batch, tag).text = value

tree.write(path, encoding="unicode", xml_declaration=False)
print(f"BatchMode configured in {path}")
PY

echo
echo "== Stage 4: run the suite through PTS (produces the native result PTS can upload) =="
echo "Result identifier: $RESULT_IDENTIFIER"
if [ "$CONFIRM" != "yes" ]; then
    echo
    echo "DRY RUN — not executing. This would run:"
    echo "  TEST_RESULTS_NAME=$RESULT_IDENTIFIER phoronix-test-suite batch-benchmark local/skainet-engine-suite"
    echo "  phoronix-test-suite upload-result $RESULT_IDENTIFIER"
    echo
    echo "Re-run with --confirm (or CONFIRM_PUBLISH=yes) to actually publish to OpenBenchmarking.org."
    exit 0
fi

TEST_RESULTS_NAME="$RESULT_IDENTIFIER" \
TEST_RESULTS_DESCRIPTION="SKaiNET engine suite — full mode, published from $JSON_DIR" \
    phoronix-test-suite batch-benchmark local/skainet-engine-suite < /dev/null

echo
echo "== Stage 5: upload to OpenBenchmarking.org =="
echo "Requires a one-time 'phoronix-test-suite openbenchmarking-login' on this host."
phoronix-test-suite upload-result "$RESULT_IDENTIFIER"

echo
echo "Published: $RESULT_IDENTIFIER"
