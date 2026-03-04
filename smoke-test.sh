#!/usr/bin/env bash
#
# smoke-test.sh — Quick smoke test for loading various LLMs via SKaiNET.
#
# Supports multiple runners (kllama, kgemma, kbert) and model formats
# (GGUF, SafeTensors) via an explicit JSON config file.
#
# Usage:
#   ./smoke-test.sh                          # use smoke-models.json if present,
#                                            #   else scan ~/.lmstudio/models
#   ./smoke-test.sh --config models.json     # use custom config file
#   ./smoke-test.sh /path/to/models          # scan custom directory (legacy)
#   ./smoke-test.sh model1.gguf model2.gguf  # run specific files (legacy)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRADLE="./gradlew --no-configuration-cache"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

separator() {
  printf '%*s\n' 80 '' | tr ' ' '─'
}

# ── Runner dispatch ──────────────────────────────────────────────────
# Maps runner name → Gradle task
runner_task() {
  case "$1" in
    kllama) echo ":skainet-apps:skainet-kllama:run" ;;
    kgemma) echo ":skainet-apps:skainet-kgemma:jvmRun" ;;
    kbert)  echo ":skainet-apps:skainet-kbert-cli:run" ;;
    *)      echo "UNKNOWN"; return 1 ;;
  esac
}

# Maps runner name → compile task
runner_compile_task() {
  case "$1" in
    kllama) echo ":skainet-apps:skainet-kllama:jvmMainClasses" ;;
    kgemma) echo ":skainet-apps:skainet-kgemma:jvmMainClasses" ;;
    kbert)  echo ":skainet-apps:skainet-kbert-cli:jvmMainClasses" ;;
    *)      echo "UNKNOWN"; return 1 ;;
  esac
}

# Builds Gradle args string based on the runner type
runner_args() {
  local runner="$1" model="$2" prompt="$3" steps="$4" temp="$5" doc="${6:-}"

  case "$runner" in
    kllama) echo "-m ${model} -s ${steps} -k ${temp} ${prompt}" ;;
    kgemma) echo "${model} \"${prompt}\" ${steps} ${temp}" ;;
    kbert)
      if [[ -n "$doc" ]]; then
        echo "${model} \"${prompt}\" \"${doc}\""
      else
        echo "${model} \"${prompt}\""
      fi
      ;;
  esac
}

# Expand ~ to $HOME in a path
expand_path() {
  local p="$1"
  if [[ "$p" == "~/"* ]]; then
    echo "${HOME}/${p#\~/}"
  else
    echo "$p"
  fi
}

# ── Parse arguments ──────────────────────────────────────────────────
CONFIG_FILE=""
LEGACY_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    *)
      LEGACY_ARGS+=("$1")
      shift
      ;;
  esac
done

# ── Determine mode: JSON config vs legacy folder scan ────────────────
USE_CONFIG=false

if [[ -n "$CONFIG_FILE" ]]; then
  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}Config file not found: ${CONFIG_FILE}${RESET}"
    exit 1
  fi
  USE_CONFIG=true
elif [[ ${#LEGACY_ARGS[@]} -eq 0 && -f "${SCRIPT_DIR}/smoke-models.json" ]]; then
  CONFIG_FILE="${SCRIPT_DIR}/smoke-models.json"
  USE_CONFIG=true
fi

# ── JSON config mode ─────────────────────────────────────────────────
if [[ "$USE_CONFIG" == true ]]; then

  # Parse model count
  MODEL_COUNT=$(python3 -c "
import json, sys
cfg = json.load(open('${CONFIG_FILE}'))
print(len(cfg['models']))
")

  # Parse defaults
  eval "$(python3 -c "
import json, sys
cfg = json.load(open('${CONFIG_FILE}'))
d = cfg.get('defaults', {})
print(f'DEF_PROMPT={repr(d.get(\"prompt\", \"The capital of France is\"))}')
print(f'DEF_STEPS={d.get(\"steps\", 32)}')
print(f'DEF_TEMP={d.get(\"temperature\", 0.0)}')
")"

  echo -e "${BOLD}SKaiNET Smoke Test${RESET} (config: $(basename "$CONFIG_FILE"))"
  echo -e "Models: ${CYAN}${MODEL_COUNT}${RESET}"
  echo -e "Default prompt:       \"${DEF_PROMPT}\""
  echo -e "Default steps:        ${DEF_STEPS}"
  echo -e "Default temperature:  ${DEF_TEMP}"
  separator

  # ── Collect unique runners and compile ───────────────────────────
  RUNNERS=$(python3 -c "
import json
cfg = json.load(open('${CONFIG_FILE}'))
print(' '.join(sorted(set(m['runner'] for m in cfg['models']))))
")

  echo -e "${YELLOW}Compiling runners: ${RUNNERS}...${RESET}"
  for runner in $RUNNERS; do
    compile_task=$(runner_compile_task "$runner")
    if ! $GRADLE "$compile_task" --quiet 2>&1; then
      echo -e "${RED}Compilation failed for ${runner}.${RESET}"
      exit 1
    fi
  done
  echo -e "${GREEN}Compilation OK${RESET}"
  separator

  # ── Run each model ───────────────────────────────────────────────
  declare -a results=()
  pass=0
  fail=0

  for i in $(seq 0 $((MODEL_COUNT - 1))); do
    # Extract model entry fields
    eval "$(python3 -c "
import json
cfg = json.load(open('${CONFIG_FILE}'))
d = cfg.get('defaults', {})
m = cfg['models'][$i]
print(f'M_NAME={repr(m[\"name\"])}')
print(f'M_RUNNER={repr(m[\"runner\"])}')
print(f'M_MODEL={repr(m[\"model\"])}')
print(f'M_FORMAT={repr(m.get(\"format\", \"gguf\"))}')
print(f'M_PROMPT={repr(m.get(\"prompt\", d.get(\"prompt\", \"The capital of France is\")))}')
print(f'M_STEPS={m.get(\"steps\", d.get(\"steps\", 32))}')
print(f'M_TEMP={m.get(\"temperature\", d.get(\"temperature\", 0.0))}')
print(f'M_DOC={repr(m.get(\"doc\", \"\"))}')
")"

    M_MODEL=$(expand_path "$M_MODEL")

    echo -e "\n${BOLD}Model:${RESET}  $M_NAME"
    echo -e "${BOLD}Runner:${RESET} $M_RUNNER  Format: $M_FORMAT"
    echo -e "${BOLD}Path:${RESET}   $M_MODEL"

    # Check model path exists
    if [[ ! -e "$M_MODEL" ]]; then
      echo -e "  ${RED}FAIL${RESET} (model path not found)"
      fail=$((fail + 1))
      results+=("FAIL|$M_NAME|$M_RUNNER|-|-|not found")
      separator
      continue
    fi

    # Determine size
    if [[ -f "$M_MODEL" ]]; then
      model_size=$(du -h "$M_MODEL" | cut -f1 | xargs)
    else
      model_size=$(du -sh "$M_MODEL" | cut -f1 | xargs)
    fi

    task=$(runner_task "$M_RUNNER")
    args=$(runner_args "$M_RUNNER" "$M_MODEL" "$M_PROMPT" "$M_STEPS" "$M_TEMP" "$M_DOC")

    start_ts=$(python3 -c 'import time; print(time.time())')
    output_file=$(mktemp)
    exit_code=0

    $GRADLE "$task" --quiet --args="$args" \
      > "$output_file" 2>&1 || exit_code=$?

    end_ts=$(python3 -c 'import time; print(time.time())')
    wall_sec=$(python3 -c "print(f'{$end_ts - $start_ts:.1f}')")

    if [[ $exit_code -ne 0 ]]; then
      echo -e "  ${RED}FAIL${RESET} (exit $exit_code, wall ${wall_sec}s)"
      tail -5 "$output_file" | sed 's/^/  │ /'
      fail=$((fail + 1))
      results+=("FAIL|$M_NAME|$M_RUNNER|$model_size|-|${wall_sec}s")
    else
      tps=$(grep -oE 'tok/s: [0-9.]+' "$output_file" | grep -oE '[0-9.]+' | tail -1)
      tps=${tps:-"?"}
      echo -e "  ${GREEN}OK${RESET}   tok/s: ${CYAN}${tps}${RESET}  wall: ${wall_sec}s"
      sed -n '/^---$/,/^---$/p' "$output_file" | grep -v '^---$' | head -3 | sed 's/^/  │ /'
      pass=$((pass + 1))
      results+=("OK|$M_NAME|$M_RUNNER|$model_size|$tps|${wall_sec}s")
    fi

    rm -f "$output_file"
    separator
  done

  # ── Summary ──────────────────────────────────────────────────────
  echo ""
  echo -e "${BOLD}Summary${RESET}"
  separator
  printf "  %-6s %-30s %-8s %8s %10s %8s\n" "Status" "Model" "Runner" "Size" "tok/s" "Wall"
  separator
  for r in "${results[@]}"; do
    IFS='|' read -r status name runner size tps wall <<< "$r"
    if [[ "$status" == "OK" ]]; then
      color="$GREEN"
    else
      color="$RED"
    fi
    printf "  ${color}%-6s${RESET} %-30s %-8s %8s %10s %8s\n" \
      "$status" "${name:0:30}" "$runner" "$size" "$tps" "$wall"
  done
  separator
  echo -e "  ${GREEN}Pass: $pass${RESET}  ${RED}Fail: $fail${RESET}  Total: ${MODEL_COUNT}"
  echo ""
  exit 0
fi

# ── Legacy fallback: folder scan / explicit .gguf files ──────────────
PROMPT="${SMOKE_PROMPT:-The capital of France is}"
STEPS="${SMOKE_STEPS:-32}"
TEMP="${SMOKE_TEMP:-0.0}"
MODEL_DIR="${LEGACY_ARGS[0]:-$HOME/.lmstudio/models}"
TASK=":skainet-apps:skainet-kllama:run"

models=()

if [[ ${#LEGACY_ARGS[@]} -gt 0 ]]; then
  for arg in "${LEGACY_ARGS[@]}"; do
    if [[ -f "$arg" && "$arg" == *.gguf ]]; then
      models+=("$arg")
    elif [[ -d "$arg" ]]; then
      while IFS= read -r -d '' f; do
        models+=("$f")
      done < <(find "$arg" -name '*.gguf' -type f -print0 2>/dev/null)
    fi
  done
else
  while IFS= read -r -d '' f; do
    models+=("$f")
  done < <(find "$MODEL_DIR" -name '*.gguf' -type f -print0 2>/dev/null)
fi

if [[ ${#models[@]} -eq 0 ]]; then
  echo -e "${RED}No .gguf models found.${RESET}"
  echo "Usage: $0 [--config models.json] [model-dir-or-files...]"
  exit 1
fi

# Sort by file size (smallest first for faster feedback)
IFS=$'\n' models=($(for m in "${models[@]}"; do
  sz=$(stat -f%z "$m" 2>/dev/null || stat -c%s "$m" 2>/dev/null || echo 0)
  echo "$sz $m"
done | sort -n | cut -d' ' -f2-))
unset IFS

echo -e "${BOLD}SKaiNET Smoke Test${RESET} (legacy mode)"
echo -e "Models found: ${CYAN}${#models[@]}${RESET}"
echo -e "Prompt:       \"${PROMPT}\""
echo -e "Steps:        ${STEPS}"
echo -e "Temperature:  ${TEMP}"
separator

# ── Ensure project compiles ────────────────────────────────────────────
echo -e "${YELLOW}Compiling kllama (JVM)...${RESET}"
if ! $GRADLE :skainet-apps:skainet-kllama:jvmMainClasses --quiet 2>&1; then
  echo -e "${RED}Compilation failed.${RESET}"
  exit 1
fi
echo -e "${GREEN}Compilation OK${RESET}"
separator

# ── Results table ──────────────────────────────────────────────────────
declare -a results=()
pass=0
fail=0

for model in "${models[@]}"; do
  model_name=$(basename "$model")
  model_size=$(du -h "$model" | cut -f1 | xargs)

  echo -e "\n${BOLD}Model:${RESET} $model_name ($model_size)"
  echo -e "${BOLD}Path:${RESET}  $model"

  start_ts=$(python3 -c 'import time; print(time.time())')

  output_file=$(mktemp)
  exit_code=0

  $GRADLE "$TASK" --quiet --args="-m ${model} -s ${STEPS} -k ${TEMP} ${PROMPT}" \
    > "$output_file" 2>&1 || exit_code=$?

  end_ts=$(python3 -c 'import time; print(time.time())')
  wall_sec=$(python3 -c "print(f'{$end_ts - $start_ts:.1f}')")

  if [[ $exit_code -ne 0 ]]; then
    echo -e "  ${RED}FAIL${RESET} (exit $exit_code, wall ${wall_sec}s)"
    tail -5 "$output_file" | sed 's/^/  │ /'
    fail=$((fail + 1))
    results+=("FAIL|$model_name|$model_size|-|${wall_sec}s")
  else
    tps=$(grep -oE 'tok/s: [0-9.]+' "$output_file" | grep -oE '[0-9.]+' | tail -1)
    tps=${tps:-"?"}
    echo -e "  ${GREEN}OK${RESET}   tok/s: ${CYAN}${tps}${RESET}  wall: ${wall_sec}s"
    sed -n '/^---$/,/^---$/p' "$output_file" | grep -v '^---$' | head -3 | sed 's/^/  │ /'
    pass=$((pass + 1))
    results+=("OK|$model_name|$model_size|$tps|${wall_sec}s")
  fi

  rm -f "$output_file"
  separator
done

# ── Summary ────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}Summary${RESET}"
separator
printf "  %-6s %-40s %8s %10s %8s\n" "Status" "Model" "Size" "tok/s" "Wall"
separator
for r in "${results[@]}"; do
  IFS='|' read -r status name size tps wall <<< "$r"
  if [[ "$status" == "OK" ]]; then
    color="$GREEN"
  else
    color="$RED"
  fi
  printf "  ${color}%-6s${RESET} %-40s %8s %10s %8s\n" "$status" "${name:0:40}" "$size" "$tps" "$wall"
done
separator
echo -e "  ${GREEN}Pass: $pass${RESET}  ${RED}Fail: $fail${RESET}  Total: ${#models[@]}"
echo ""
