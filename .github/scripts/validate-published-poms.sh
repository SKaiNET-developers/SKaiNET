#!/usr/bin/env bash
# Fails the build if any published SKaiNET POM contains invalid coordinates.
#
# Catches the class of bug shipped in 0.19.0, where `skainet-backend-cpu`'s POM
# declared `sk.ainet:skainet-backend-api-jvm:unspecified` because
# `skainet-backend-api` was not configured to publish and the root
# `allprojects { group = "sk.ainet" }` disagreed with `GROUP=sk.ainet.core`.
#
# Two checks per generated POM under ~/.m2/repository/sk/ainet/core/**:
#   1. No `<version>unspecified</version>` anywhere in the POM.
#   2. Every `<dependency>` whose `<artifactId>` starts with `skainet-` uses
#      `<groupId>sk.ainet.core</groupId>` — `project(...)` deps on sibling
#      modules must resolve to the same publish group.

set -euo pipefail

REPO_ROOT="${HOME}/.m2/repository/sk/ainet/core"

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "ERROR: no published artifacts found under ${REPO_ROOT}" >&2
  echo "Did ./gradlew publishToMavenLocal run successfully?" >&2
  exit 1
fi

mapfile -t POMS < <(find "${REPO_ROOT}" -type f -name '*.pom' | sort)

if [[ ${#POMS[@]} -eq 0 ]]; then
  echo "ERROR: no .pom files under ${REPO_ROOT}" >&2
  exit 1
fi

echo "Scanning ${#POMS[@]} published POMs..."

report_file="$(mktemp)"
trap 'rm -f "${report_file}"' EXIT

for pom in "${POMS[@]}"; do
  rel="${pom#${REPO_ROOT}/}"

  if grep -Fq '<version>unspecified</version>' "${pom}"; then
    {
      echo "FAIL  ${rel}: contains <version>unspecified</version>"
      grep -n '<version>unspecified</version>' "${pom}" | sed 's/^/      /'
    } >> "${report_file}"
  fi

  bad_deps="$(awk '
    /<dependency>/                          { inDep=1; block=""; next }
    inDep                                   { block = block "\n" $0 }
    /<\/dependency>/ {
      inDep=0
      if (block ~ /<artifactId>skainet-/ && block !~ /<groupId>sk\.ainet\.core<\/groupId>/) {
        print block
      }
    }
  ' "${pom}")"

  if [[ -n "${bad_deps}" ]]; then
    {
      echo "FAIL  ${rel}: skainet-* dependency with non-sk.ainet.core group"
      printf '%s\n' "${bad_deps}" | sed 's/^/      /'
    } >> "${report_file}"
  fi
done

if [[ -s "${report_file}" ]]; then
  cat "${report_file}" >&2
  echo "" >&2
  echo "POM validation failed. See the 0.19.1 CHANGELOG entry for the regression this check prevents." >&2
  exit 1
fi

echo "All ${#POMS[@]} POMs look good."
