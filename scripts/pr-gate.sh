#!/usr/bin/env bash
# Local PR gate for SKaiNET: runs the same test legs as CI (.github/workflows/build.yml,
# java-tests.yml) plus the binary-compatibility check, so a PR is opened only after
# everything CI will run has passed locally. Usage:
#
#   scripts/pr-gate.sh            # full gate
#   scripts/pr-gate.sh --bench    # full gate + StorageBenchmarks and JMH microbenchmarks
#   scripts/pr-gate.sh --quick    # JVM leg + apiCheck only (iterate fast, then run the full gate)
#
# Set JAVA_HOME to a JDK 25 (CI uses 25; the build requires >= 21).
set -euo pipefail
cd "$(dirname "$0")/.."

GRADLE=(./gradlew --no-daemon --stacktrace -Dorg.gradle.caching=true -Dorg.gradle.configuration-cache=true)
mode="${1:-full}"

# Karma's ChromeHeadless (jsBrowserTest / wasmJsBrowserTest) needs a Chrome/Chromium binary.
# GitHub runners ship one; locally, point CHROME_BIN at whatever is installed.
if [[ -z "${CHROME_BIN:-}" ]]; then
  for c in google-chrome google-chrome-stable chromium chromium-browser; do
    if command -v "$c" >/dev/null 2>&1; then export CHROME_BIN="$(command -v "$c")"; break; fi
  done
fi
echo "pr-gate: JAVA_HOME=${JAVA_HOME:-<default>} CHROME_BIN=${CHROME_BIN:-<none: browser tests will fail>}"

step() { echo; echo "=== pr-gate: $* ==="; }

step "JVM tests"
"${GRADLE[@]}" jvmTest

step "binary-compatibility check (apiCheck; run 'apiDump' and commit the dumps if the change is additive)"
"${GRADLE[@]}" apiCheck

if [[ "$mode" == "--quick" ]]; then
  echo; echo "pr-gate: quick mode done — run the full gate before opening the PR."; exit 0
fi

step "JS / Wasm tests"
"${GRADLE[@]}" verifyNpmPins jsTest wasmJsTest wasmWasiTest

step "Kotlin/Native linuxX64 tests"
"${GRADLE[@]}" linuxX64Test

step "assemble"
"${GRADLE[@]}" assemble

step "Java consumer API tests"
"${GRADLE[@]}" :skainet-test:skainet-test-java:test

if [[ "$mode" == "--bench" ]]; then
  step "benchmarks (compare against the committed baseline before/after)"
  "${GRADLE[@]}" :skainet-lang:skainet-lang-core:jvmBenchmark
  "${GRADLE[@]}" :skainet-backends:benchmarks:jvm-cpu-jmh:jmh
fi

echo; echo "pr-gate: all legs passed."
