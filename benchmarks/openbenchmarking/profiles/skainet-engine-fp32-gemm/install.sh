#!/usr/bin/env bash
# PTS install hook: invoked with $1 = install destination directory.
# Drops a wrapper script that forwards to the SKaiNET publish harness jar.
# Caller (scripts/run_engine_benchmarks.sh) must set SKAINET_PUBLISH_JAR.
set -euo pipefail

DEST="$1"
SCENARIO="engine-fp32-gemm"

if [ -z "${SKAINET_PUBLISH_JAR:-}" ]; then
    echo "ERROR: SKAINET_PUBLISH_JAR not set. Build :skainet-backends:benchmarks:jvm-cpu-publish:shadowJar and export its absolute path." >&2
    exit 2
fi
if [ ! -f "$SKAINET_PUBLISH_JAR" ]; then
    echo "ERROR: SKAINET_PUBLISH_JAR not found: $SKAINET_PUBLISH_JAR" >&2
    exit 2
fi

cp "$SKAINET_PUBLISH_JAR" "$DEST/skainet-engine-publish.jar"

cat > "$DEST/$SCENARIO" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec java --enable-preview --add-modules jdk.incubator.vector \
    -jar "$HERE/skainet-engine-publish.jar" \
    run --scenario engine-fp32-gemm --out "$HERE/last-result.json" "$@"
WRAPPER
chmod +x "$DEST/$SCENARIO"
