#!/usr/bin/env bash
# Run the full SKaiNET compute-engine benchmark suite on this host.
# Writes one JSON record per (scenario, provider) under out/engine/<timestamp>/.
#
# Usage:
#   scripts/run_engine_benchmarks.sh [--smoke] [--out-dir DIR]
#
# Flags:
#   --smoke     Use the smoke run shape (1 warmup + 1 measured + smallest shape).
#   --out-dir   Override the default output directory.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE=false
OUT_BASE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --smoke) SMOKE=true; shift ;;
        --out-dir) OUT_BASE="$2"; shift 2 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_BASE:-$REPO_ROOT/out/engine/$TIMESTAMP}"
mkdir -p "$OUT_DIR"

# CPU governor warning. The full lane benefits from a fixed performance
# governor; on a laptop this is rarely set by default. Don't fail — just
# warn so the operator can fix it before publishing.
if [ -r /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
    if [ "$GOV" != "performance" ] && [ "$SMOKE" != "true" ]; then
        echo "WARNING: cpu0 governor is '$GOV', not 'performance'. Results may be noisy." >&2
        echo "         sudo cpupower frequency-set -g performance   # to fix" >&2
    fi
fi

echo "Building publication harness shadow jar..."
./gradlew --no-daemon :skainet-backends:benchmarks:jvm-cpu-publish:shadowJar -q

JAR="$(ls -t skainet-backends/benchmarks/jvm-cpu-publish/build/libs/skainet-engine-publish-*-all.jar | head -n1)"
if [ ! -f "$JAR" ]; then
    echo "ERROR: shadow jar not found after build" >&2
    exit 1
fi
JAR="$(cd "$(dirname "$JAR")" && pwd)/$(basename "$JAR")"
export SKAINET_PUBLISH_JAR="$JAR"

MANIFEST="$REPO_ROOT/benchmarks/manifests/engine-release.yml"
echo "Manifest: $MANIFEST"
echo "Output:   $OUT_DIR"
echo "Mode:     $([ "$SMOKE" = "true" ] && echo smoke || echo full)"
echo "Jar:      $JAR"
echo

# Parse manifest with a minimal awk script — yq is not assumed installed
# on stock self-hosted runners. We only need the scenarios list.
SCENARIOS_TSV="$(awk '
    /^scenarios:/ { in_s = 1; next }
    in_s && /^[a-zA-Z]/ { in_s = 0 }
    in_s && /^  - id:/ { id = $3; provider = "" }
    in_s && /^    provider:/ { provider = $2; if (id && provider) print id "\t" provider; id = ""; provider = "" }
' "$MANIFEST")"

if [ -z "$SCENARIOS_TSV" ]; then
    echo "ERROR: could not parse scenarios from $MANIFEST" >&2
    exit 1
fi

if [ "$SMOKE" = "true" ]; then
    SMOKE_FLAG="--smoke"
else
    SMOKE_FLAG=""
fi

JAVA_VM_FLAGS=(--enable-preview --add-modules jdk.incubator.vector)

while IFS=$'\t' read -r SCN PROV; do
    [ -z "$SCN" ] && continue
    OUT="$OUT_DIR/${SCN}-${PROV}.json"
    echo "== $SCN (provider=$PROV) -> $OUT"
    java "${JAVA_VM_FLAGS[@]}" -jar "$JAR" run \
        --scenario "$SCN" \
        --provider "$PROV" \
        --out "$OUT" \
        $SMOKE_FLAG
done <<< "$SCENARIOS_TSV"

echo
echo "Done. Records under: $OUT_DIR"
ls -1 "$OUT_DIR"
