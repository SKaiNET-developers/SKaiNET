#!/usr/bin/env bash
# Idempotently create/update the GitHub labels declared in .github/labels.txt.
#
# Usage:
#   .github/scripts/sync-labels.sh                       # against the current repo
#   .github/scripts/sync-labels.sh -R owner/repo         # against another repo
#   DRY_RUN=1 .github/scripts/sync-labels.sh             # print what would run
#
# Requires `gh` authenticated with triage (or higher) permission on the repo.
# Existing labels are updated in place (--force); labels that are not in
# labels.txt are left untouched — this script never deletes anything.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
labels_file="${LABELS_FILE:-$here/../labels.txt}"
repo_args=("$@")

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI not found" >&2
  exit 1
fi

count=0
while IFS='|' read -r name color description; do
  # skip comments and blank lines
  case "$name" in ''|\#*) continue ;; esac
  name="${name%"${name##*[![:space:]]}"}"          # rtrim
  color="${color//[[:space:]]/}"
  description="${description#"${description%%[![:space:]]*}"}"  # ltrim
  cmd=(gh label create "$name" --color "$color" --description "$description" --force "${repo_args[@]}")
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
  count=$((count + 1))
done < "$labels_file"

echo "synced $count labels from $labels_file"
