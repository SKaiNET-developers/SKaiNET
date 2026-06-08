#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="${MINERVA_RUNTIME_ROOT:-${1:-}}"

if [[ -z "${RUNTIME_ROOT}" ]]; then
  echo "Usage: MINERVA_RUNTIME_ROOT=/path/to/libminerva $0" >&2
  echo "       or: $0 /path/to/libminerva" >&2
  exit 2
fi

RUNTIME_ROOT="$(cd "${RUNTIME_ROOT}" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
COMPILER_SCRIPT="${MINERVA_COMPILER_SCRIPT:-${RUNTIME_ROOT}/compiler/minerva_compile.py}"
PROFILE_DIR="${MINERVA_PROFILE_DIR:-${ROOT_DIR}/build/minerva-real-runtime-profile}"
PROJECT_DIR="${ROOT_DIR}/build/minerva/TinySecureMlp"
KEY_FILE="${MINERVA_KEY_FILE:-${PROFILE_DIR}/device.key}"
CALIBRATION_NPZ="${MINERVA_CALIBRATION_NPZ:-${PROFILE_DIR}/calibration.npz}"
SECRET_INCLUDE_DIR="${PROFILE_DIR}/secret-include"
SECRET_HEADER="${SECRET_INCLUDE_DIR}/secrets.h"
AVR_PGMSPACE_HEADER="${SECRET_INCLUDE_DIR}/avr/pgmspace.h"
HOST_ADAPTER_SOURCE="${MINERVA_HOST_ADAPTER_SOURCE:-${PROJECT_DIR}/host/runtime_adapter.example.c}"
HOST_TOLERANCE="${MINERVA_HOST_TOLERANCE:-1.0}"

mkdir -p "${PROFILE_DIR}" "${SECRET_INCLUDE_DIR}" "$(dirname "${AVR_PGMSPACE_HEADER}")"

if [[ ! -f "${COMPILER_SCRIPT}" ]]; then
  echo "Minerva compiler script not found: ${COMPILER_SCRIPT}" >&2
  exit 2
fi

if [[ ! -f "${KEY_FILE}" ]]; then
  "${PYTHON_BIN}" "${COMPILER_SCRIPT}" --gen-key "${KEY_FILE}"
fi

"${PYTHON_BIN}" - "${CALIBRATION_NPZ}" <<'PY'
from pathlib import Path
import sys
import numpy as np

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
samples = np.array(
    [
        [0.25, 0.50, 0.75, 1.00],
        [0.10, 0.25, 0.50, 0.75],
        [-0.25, 0.00, 0.25, 0.50],
        [1.00, 0.75, 0.50, 0.25],
    ],
    dtype=np.float32,
)
np.savez(path, X=samples)
PY

"${PYTHON_BIN}" - "${KEY_FILE}" "${SECRET_HEADER}" <<'PY'
from pathlib import Path
import sys

key_path = Path(sys.argv[1])
header_path = Path(sys.argv[2])
key = key_path.read_bytes()
if len(key) < 32:
    raise SystemExit(f"Minerva key must contain at least 32 bytes: {key_path}")
values = ", ".join(f"0x{byte:02X}" for byte in key[:32])
header_path.parent.mkdir(parents=True, exist_ok=True)
header_path.write_text(
    "#pragma once\n"
    "#include <stdint.h>\n"
    "static const uint8_t skainet_minerva_host_key[32] = { "
    + values
    + " };\n"
    "#define MNV_DEVICE_KEY skainet_minerva_host_key\n"
)
PY

cat > "${AVR_PGMSPACE_HEADER}" <<'EOF'
#pragma once
#include <string.h>

#ifndef PROGMEM
#define PROGMEM
#endif

#ifndef memcpy_P
#define memcpy_P(destination, source, size) memcpy((destination), (source), (size))
#endif

#ifndef pgm_read_byte
#define pgm_read_byte(address) (*(const unsigned char *)(address))
#endif
EOF

cd "${ROOT_DIR}"

./gradlew :skainet-compile:skainet-compile-minerva:minervaHostVerification \
  -Pminerva.hostVerification.enabled=true \
  -Pminerva.runtimeRoot="${RUNTIME_ROOT}" \
  -Pminerva.compilerScript="${COMPILER_SCRIPT}" \
  -Pminerva.keyFile="${KEY_FILE}" \
  -Pminerva.calibrationNpz="${CALIBRATION_NPZ}" \
  -Pminerva.hostVerification.tolerance="${HOST_TOLERANCE}" \
  -Pminerva.hostVerification.hostAdapterSource="${HOST_ADAPTER_SOURCE}" \
  -Pminerva.hostVerification.hostIncludeDirs="${SECRET_INCLUDE_DIR}" \
  -Pminerva.hostVerification.runCmakeBuild=true \
  -Pminerva.hostVerification.runCTest=true

echo "Minerva real-runtime profile completed."
echo "Bundle: ${PROJECT_DIR}"
echo "Profile artifacts: ${PROFILE_DIR}"
