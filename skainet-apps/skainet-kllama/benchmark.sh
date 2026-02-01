#!/bin/bash
#
# KLlama Backend Benchmark Script
#
# Builds and benchmarks kllama across different execution backends:
# - JVM Default CPU (Vector API disabled)
# - JVM Vector API (SIMD accelerated)
# - Native macOS CPU
# - Native macOS MLX (GPU accelerated)
#
# Usage: ./benchmark.sh <model.gguf> [prompt] [steps]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
MODEL_PATH="${1:-}"
PROMPT="${2:-Once upon a time}"
STEPS="${3:-64}"
WARMUP_STEPS=10
BENCHMARK_RUNS=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Output paths
JVM_JAR="$SCRIPT_DIR/build/libs/kllama-all.jar"
NATIVE_EXEC="$SCRIPT_DIR/build/bin/macosArm64/releaseExecutable/kllama.kexe"

# Results storage (bash 3.x compatible)
RESULT_JVM_SCALAR=""
RESULT_JVM_VECTOR=""
RESULT_NATIVE_CPU=""
RESULT_NATIVE_MLX=""

print_header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║           KLlama Backend Benchmark Suite                     ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_usage() {
    echo "Usage: $0 <model.gguf> [prompt] [steps]"
    echo ""
    echo "Arguments:"
    echo "  model.gguf    Path to GGUF model file (required)"
    echo "  prompt        Text prompt for generation (default: 'Once upon a time')"
    echo "  steps         Number of tokens to generate (default: 64)"
    echo ""
    echo "Environment variables:"
    echo "  MLX_ROOT      Path to MLX installation (default: /opt/homebrew/opt/mlx)"
    echo "  SKIP_BUILD    Set to 'true' to skip build step"
    echo ""
    echo "Examples:"
    echo "  $0 tinyllama.gguf"
    echo "  $0 tinyllama.gguf \"Hello world\" 128"
    echo "  SKIP_BUILD=true $0 tinyllama.gguf"
}

check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"

    # Check model file
    if [[ -z "$MODEL_PATH" ]]; then
        echo -e "${RED}Error: Model path is required${NC}"
        print_usage
        exit 1
    fi

    if [[ ! -f "$MODEL_PATH" ]]; then
        echo -e "${RED}Error: Model file not found: $MODEL_PATH${NC}"
        exit 1
    fi

    # Check Java
    if ! command -v java &> /dev/null; then
        echo -e "${RED}Error: Java not found. JDK 21+ is required.${NC}"
        exit 1
    fi

    JAVA_VERSION=$(java -version 2>&1 | head -1 | cut -d'"' -f2 | cut -d'.' -f1)
    if [[ "$JAVA_VERSION" -lt 21 ]]; then
        echo -e "${YELLOW}Warning: Java $JAVA_VERSION detected. JDK 21+ recommended for Vector API.${NC}"
    else
        echo -e "${GREEN}✓ Java $JAVA_VERSION detected${NC}"
    fi

    # Check MLX
    MLX_ROOT="${MLX_ROOT:-/opt/homebrew/opt/mlx}"
    if [[ -d "$MLX_ROOT/include/mlx" ]]; then
        echo -e "${GREEN}✓ MLX found at $MLX_ROOT${NC}"
        HAS_MLX=true
    else
        echo -e "${YELLOW}⚠ MLX not found at $MLX_ROOT. MLX benchmark will be skipped.${NC}"
        HAS_MLX=false
    fi

    # Check platform
    if [[ "$(uname -m)" != "arm64" ]]; then
        echo -e "${YELLOW}⚠ Not running on Apple Silicon. Native benchmarks may not work.${NC}"
    fi

    echo ""
}

build_targets() {
    if [[ "${SKIP_BUILD:-false}" == "true" ]]; then
        echo -e "${YELLOW}Skipping build (SKIP_BUILD=true)${NC}"
        return
    fi

    echo -e "${BLUE}Building targets...${NC}"
    cd "$PROJECT_ROOT"

    # Build JVM fat JAR
    echo -e "${CYAN}Building JVM fat JAR...${NC}"
    ./gradlew :skainet-apps:skainet-kllama:shadowJar --quiet --no-configuration-cache

    if [[ -f "$JVM_JAR" ]]; then
        echo -e "${GREEN}✓ JVM JAR built: $JVM_JAR${NC}"
    else
        echo -e "${RED}✗ JVM JAR build failed${NC}"
        exit 1
    fi

    # Build native macOS executable
    echo -e "${CYAN}Building native macOS executable...${NC}"
    if [[ "$HAS_MLX" == "true" ]]; then
        MLX_ROOT="$MLX_ROOT" ./gradlew :skainet-apps:skainet-kllama:linkReleaseExecutableMacosArm64 --quiet --no-configuration-cache
    else
        ./gradlew :skainet-apps:skainet-kllama:linkReleaseExecutableMacosArm64 --quiet --no-configuration-cache
    fi

    if [[ -f "$NATIVE_EXEC" ]]; then
        echo -e "${GREEN}✓ Native executable built: $NATIVE_EXEC${NC}"
    else
        echo -e "${RED}✗ Native executable build failed${NC}"
        exit 1
    fi

    echo ""
}

# Run a single benchmark and return the average tok/s
# Args: $1 = command to run, $2 = number of runs
run_single_benchmark() {
    local cmd="$1"
    local runs="${2:-$BENCHMARK_RUNS}"

    local total_toks=0
    local valid_runs=0

    for ((i=1; i<=runs; i++)); do
        # Capture output and extract tok/s
        local output
        output=$(eval "$cmd" 2>&1) || true

        # Extract tok/s from output (looking for "tok/s: X.XX" pattern)
        local toks_per_sec
        toks_per_sec=$(echo "$output" | grep -oE 'tok/s: [0-9]+\.?[0-9]*' | grep -oE '[0-9]+\.?[0-9]*' | tail -1)

        if [[ -n "$toks_per_sec" ]]; then
            echo -ne "  Run $i/$runs: ${GREEN}$toks_per_sec tok/s${NC}\n"
            total_toks=$(echo "$total_toks + $toks_per_sec" | bc)
            ((valid_runs++)) || true
        else
            echo -ne "  Run $i/$runs: ${RED}Failed${NC}\n"
        fi
    done

    # Calculate and return average
    if [[ $valid_runs -gt 0 ]]; then
        echo "scale=2; $total_toks / $valid_runs" | bc
    else
        echo "N/A"
    fi
}

run_jvm_cpu_benchmark() {
    echo -e "${BLUE}━━━ JVM Default CPU (Vector API disabled) ━━━${NC}"

    # Warmup
    echo -e "  Warming up..."
    SKAINET_CPU_VECTOR_ENABLED=false java -jar "$JVM_JAR" "$MODEL_PATH" "$PROMPT" "$STEPS" 0.0 > /dev/null 2>&1 || true

    local cmd="SKAINET_CPU_VECTOR_ENABLED=false java -jar \"$JVM_JAR\" \"$MODEL_PATH\" \"$PROMPT\" $STEPS 0.0"
    RESULT_JVM_SCALAR=$(run_single_benchmark "$cmd")
    echo -e "  ${GREEN}Average: $RESULT_JVM_SCALAR tok/s${NC}"
    echo ""
}

run_jvm_vector_benchmark() {
    echo -e "${BLUE}━━━ JVM Vector API (SIMD) ━━━${NC}"

    # Warmup
    echo -e "  Warming up..."
    SKAINET_CPU_VECTOR_ENABLED=true java --enable-preview --add-modules jdk.incubator.vector -jar "$JVM_JAR" "$MODEL_PATH" "$PROMPT" "$STEPS" 0.0 > /dev/null 2>&1 || true

    local cmd="SKAINET_CPU_VECTOR_ENABLED=true java --enable-preview --add-modules jdk.incubator.vector -jar \"$JVM_JAR\" \"$MODEL_PATH\" \"$PROMPT\" $STEPS 0.0"
    RESULT_JVM_VECTOR=$(run_single_benchmark "$cmd")
    echo -e "  ${GREEN}Average: $RESULT_JVM_VECTOR tok/s${NC}"
    echo ""
}

run_native_cpu_benchmark() {
    echo -e "${BLUE}━━━ Native macOS CPU ━━━${NC}"

    # Warmup
    echo -e "  Warming up..."
    "$NATIVE_EXEC" "$MODEL_PATH" "$PROMPT" --steps "$STEPS" --temp 0.0 --backend cpu > /dev/null 2>&1 || true

    local cmd="\"$NATIVE_EXEC\" \"$MODEL_PATH\" \"$PROMPT\" --steps $STEPS --temp 0.0 --backend cpu"
    RESULT_NATIVE_CPU=$(run_single_benchmark "$cmd")
    echo -e "  ${GREEN}Average: $RESULT_NATIVE_CPU tok/s${NC}"
    echo ""
}

run_native_mlx_benchmark() {
    if [[ "$HAS_MLX" != "true" ]]; then
        echo -e "${YELLOW}━━━ Native macOS MLX (Skipped - MLX not available) ━━━${NC}"
        RESULT_NATIVE_MLX="N/A"
        echo ""
        return
    fi

    echo -e "${BLUE}━━━ Native macOS MLX (GPU) ━━━${NC}"

    # Warmup
    echo -e "  Warming up..."
    "$NATIVE_EXEC" "$MODEL_PATH" "$PROMPT" --steps "$STEPS" --temp 0.0 --backend mlx > /dev/null 2>&1 || true

    local cmd="\"$NATIVE_EXEC\" \"$MODEL_PATH\" \"$PROMPT\" --steps $STEPS --temp 0.0 --backend mlx"
    RESULT_NATIVE_MLX=$(run_single_benchmark "$cmd")
    echo -e "  ${GREEN}Average: $RESULT_NATIVE_MLX tok/s${NC}"
    echo ""
}

print_results() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                    Benchmark Results                         ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "Model: $MODEL_PATH"
    echo -e "Prompt: \"$PROMPT\""
    echo -e "Steps: $STEPS"
    echo -e "Runs: $BENCHMARK_RUNS"
    echo ""

    printf "%-25s %15s %15s\n" "Backend" "tok/s" "Speedup"
    printf "%-25s %15s %15s\n" "-------------------------" "---------------" "---------------"

    # Get baseline (JVM CPU Scalar)
    local baseline="$RESULT_JVM_SCALAR"
    if [[ "$baseline" == "N/A" || -z "$baseline" ]]; then
        baseline=1
    fi

    # Print JVM Scalar
    local speedup="1.00x"
    printf "%-25s %15s %15s\n" "JVM CPU (Scalar)" "$RESULT_JVM_SCALAR" "$speedup"

    # Print JVM Vector
    if [[ "$RESULT_JVM_VECTOR" != "N/A" && -n "$RESULT_JVM_VECTOR" ]]; then
        speedup=$(echo "scale=2; $RESULT_JVM_VECTOR / $baseline" | bc)
        speedup="${speedup}x"
    else
        speedup="N/A"
    fi
    printf "%-25s %15s %15s\n" "JVM Vector API" "$RESULT_JVM_VECTOR" "$speedup"

    # Print Native CPU
    if [[ "$RESULT_NATIVE_CPU" != "N/A" && -n "$RESULT_NATIVE_CPU" ]]; then
        speedup=$(echo "scale=2; $RESULT_NATIVE_CPU / $baseline" | bc)
        speedup="${speedup}x"
    else
        speedup="N/A"
    fi
    printf "%-25s %15s %15s\n" "Native CPU" "$RESULT_NATIVE_CPU" "$speedup"

    # Print Native MLX
    if [[ "$RESULT_NATIVE_MLX" != "N/A" && -n "$RESULT_NATIVE_MLX" ]]; then
        speedup=$(echo "scale=2; $RESULT_NATIVE_MLX / $baseline" | bc)
        speedup="${speedup}x"
    else
        speedup="N/A"
    fi
    printf "%-25s %15s %15s\n" "Native MLX" "$RESULT_NATIVE_MLX" "$speedup"

    echo ""
    echo -e "${GREEN}Benchmark complete!${NC}"
}

# Main execution
main() {
    print_header
    check_prerequisites
    build_targets

    echo -e "${BLUE}Starting benchmarks...${NC}"
    echo -e "Using temperature=0.0 (greedy sampling) for deterministic results"
    echo ""

    run_jvm_cpu_benchmark
    run_jvm_vector_benchmark
    run_native_cpu_benchmark
    run_native_mlx_benchmark

    print_results
}

main "$@"
