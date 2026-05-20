#!/usr/bin/env bash
# Checks that the start-path docs reference the same SKaiNET version as the
# README. The README "Start in 5 minutes" / Quickstart block is the documented
# source of truth for first-run snippets.
set -euo pipefail

cd "$(dirname "$0")/.."

readme_version="$(grep -oE 'sk\.ainet:skainet-bom:[0-9]+\.[0-9]+\.[0-9]+' README.md \
  | head -n1 | cut -d: -f3)"

if [[ -z "${readme_version}" ]]; then
  echo "FAIL: could not find a skainet-bom version in README.md"
  exit 1
fi

echo "README source-of-truth version: ${readme_version}"

status=0
check() {
  local file="$1"
  local found
  found="$(grep -oE '[0-9]+\.[0-9]+\.[0-9]+' "${file}" | sort -u | tr '\n' ' ' || true)"
  if grep -q "${readme_version}" "${file}"; then
    echo "OK   ${file}"
  else
    echo "FAIL ${file} does not reference ${readme_version} (found: ${found})"
    status=1
  fi
}

check docs/modules/ROOT/pages/tutorials/java-getting-started.adoc

exit "${status}"
