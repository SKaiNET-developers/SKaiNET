#!/usr/bin/env bash
# Register this host as a GitHub Actions self-hosted runner for the
# SKaiNET engine benchmarks job.
#
# OPERATOR ACTION REQUIRED — this script needs a runner registration
# token, which you generate in the GitHub repo's
#   Settings → Actions → Runners → New self-hosted runner
# page. Tokens expire after ~1 hour, so generate just before running.
#
# Usage:
#   GH_RUNNER_TOKEN=AAA... REPO=owner/SKaiNET ./scripts/register_bench_runner.sh
#
# Result:
#   - Downloads the actions/runner release into ~/actions-runner
#   - Configures it with labels: self-hosted,linux,x86_64,skainet-bench-linux-x86
#   - Installs and starts a systemd service that survives reboots
set -euo pipefail

REPO="${REPO:?set REPO=owner/repo (e.g. ainet-sk/SKaiNET)}"
TOKEN="${GH_RUNNER_TOKEN:?set GH_RUNNER_TOKEN from Settings -> Actions -> Runners -> New self-hosted runner}"

RUNNER_VERSION="${RUNNER_VERSION:-2.328.0}"
RUNNER_DIR="${RUNNER_DIR:-$HOME/actions-runner}"
RUNNER_LABELS="self-hosted,linux,x86_64,skainet-bench-linux-x86"
RUNNER_NAME="${RUNNER_NAME:-$(hostname)-skainet-bench}"

mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

if [ ! -f "./config.sh" ]; then
    TARBALL="actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
    echo "Downloading $TARBALL"
    curl -fLo "$TARBALL" \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/${TARBALL}"
    tar xzf "$TARBALL"
    rm -f "$TARBALL"
fi

if [ -f ".runner" ]; then
    echo "Runner already configured at $RUNNER_DIR — remove it first with ./config.sh remove --token <token>"
    exit 1
fi

./config.sh \
    --url "https://github.com/$REPO" \
    --token "$TOKEN" \
    --name "$RUNNER_NAME" \
    --labels "$RUNNER_LABELS" \
    --unattended \
    --replace

echo
echo "Installing systemd service (requires sudo)"
sudo ./svc.sh install
sudo ./svc.sh start

echo
echo "Runner registered as: $RUNNER_NAME"
echo "Labels: $RUNNER_LABELS"
echo "Repo:   https://github.com/$REPO/settings/actions/runners"
