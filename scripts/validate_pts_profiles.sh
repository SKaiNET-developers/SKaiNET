#!/usr/bin/env bash
# Validate every local PTS profile + the engine suite via phoronix-test-suite.
# Skips with a clear message (exit 0) if PTS isn't installed, so the smoke CI
# job doesn't fail when only the JSON harness is being exercised.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROFILES_DIR="$REPO_ROOT/benchmarks/openbenchmarking/profiles"
SUITES_DIR="$REPO_ROOT/benchmarks/openbenchmarking/suites"

if ! command -v phoronix-test-suite >/dev/null 2>&1; then
    echo "phoronix-test-suite not installed — skipping PTS validation."
    echo "Install with: sudo apt-get install -y phoronix-test-suite"
    exit 0
fi

PTS_USER_HOME="${XDG_DATA_HOME:-$HOME/.phoronix-test-suite}"
LOCAL_TP_DIR="$PTS_USER_HOME/test-profiles/local"
LOCAL_TS_DIR="$PTS_USER_HOME/test-suites/local"
mkdir -p "$LOCAL_TP_DIR" "$LOCAL_TS_DIR"

echo "Syncing local profiles into $LOCAL_TP_DIR"
for prof_dir in "$PROFILES_DIR"/*/; do
    name="$(basename "$prof_dir")"
    target="$LOCAL_TP_DIR/$name"
    rm -rf "$target"
    cp -a "$prof_dir" "$target"
done

echo "Syncing local suite into $LOCAL_TS_DIR"
for suite_dir in "$SUITES_DIR"/*/; do
    name="$(basename "$suite_dir")"
    target="$LOCAL_TS_DIR/$name"
    rm -rf "$target"
    cp -a "$suite_dir" "$target"
done

FAIL=0
echo
echo "Validating test profiles..."
for prof_dir in "$PROFILES_DIR"/*/; do
    name="$(basename "$prof_dir")"
    echo "  - local/$name"
    if ! phoronix-test-suite validate-test-profile "local/$name"; then
        echo "    FAILED" >&2
        FAIL=1
    fi
done

echo
echo "Validating test suites..."
for suite_dir in "$SUITES_DIR"/*/; do
    name="$(basename "$suite_dir")"
    echo "  - local/$name"
    if ! phoronix-test-suite validate-test-suite "local/$name"; then
        echo "    FAILED" >&2
        FAIL=1
    fi
done

if [ $FAIL -ne 0 ]; then
    echo
    echo "One or more PTS assets failed validation." >&2
    exit 1
fi

echo
echo "All PTS assets validated successfully."
