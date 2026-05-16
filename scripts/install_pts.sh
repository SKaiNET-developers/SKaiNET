#!/usr/bin/env bash
# Install Phoronix Test Suite on a Linux benchmark runner. Ubuntu 24.04+
# doesn't ship PTS in the default repos, so we pull the upstream .deb and
# install its runtime dependencies via apt.
#
# Run as the operator (uses sudo). Idempotent — safe to re-run.
set -euo pipefail

PTS_VERSION="${PTS_VERSION:-10.8.6}"
DEB_URL="https://phoronix-test-suite.com/releases/repo/pts.debian/files/phoronix-test-suite_${PTS_VERSION}_all.deb"
TMP_DEB="/tmp/phoronix-test-suite_${PTS_VERSION}_all.deb"

if command -v phoronix-test-suite >/dev/null 2>&1; then
    INSTALLED="$(phoronix-test-suite version 2>/dev/null | head -1 || true)"
    echo "phoronix-test-suite already installed: $INSTALLED"
    if [ "${1:-}" != "--force" ]; then
        echo "Re-run with --force to reinstall."
        exit 0
    fi
fi

echo "Installing runtime dependencies via apt..."
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    php-cli php-xml php-gd php-zip php-bz2 \
    curl wget bzip2 unzip tar

echo "Downloading PTS $PTS_VERSION from $DEB_URL"
wget -q -O "$TMP_DEB" "$DEB_URL"

echo "Installing the .deb"
sudo apt-get install -y "$TMP_DEB"
rm -f "$TMP_DEB"

echo
phoronix-test-suite version | head -3
