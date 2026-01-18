# SKaiNET Ground Truth Testing Architecture

This document describes the testing infrastructure for validating SKaiNET tensor operations against PyTorch ground truth.

## Overview

The ground truth testing system ensures that SKaiNET operations produce identical results to PyTorch. It consists of three projects working together:

```mermaid
graph TB
    subgraph "Python/Docker"
        GT[skainet-ground-truth<br/>PyTorch Test Cases]
        TRACER[gradienttracer<br/>Execution & Serialization]
        DOCKER[Docker Container]
    end

    subgraph "Kotlin/JVM"
        VALIDATOR[skainet-test-groundtruth<br/>Validation Module]
        SKAINET[SKaiNET<br/>Tensor Operations]
    end

    subgraph "Artifacts"
        GGUF[(GGUF Files<br/>Ground Truth Data)]
    end

    GT --> TRACER
    TRACER --> DOCKER
    DOCKER --> GGUF
    GGUF --> VALIDATOR
    VALIDATOR --> SKAINET

    style GGUF fill:#f9f,stroke:#333,stroke-width:2px
```

## Projects & Responsibilities

| Project | Language | Responsibility |
|---------|----------|----------------|
| **gradienttracer** | Python | Core framework: test discovery, execution, GGUF serialization |
| **skainet-ground-truth** | Python | Test case definitions organized by test suite |
| **skainet-test-groundtruth** | Kotlin | GGUF loading, validation, SKaiNET comparison |

## Data Flow

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant Gradle as Gradle Build
    participant Docker as Docker Container
    participant PyTorch as PyTorch Runtime
    participant GGUF as GGUF Files
    participant Validator as GroundTruthValidator
    participant SKaiNET as SKaiNET Ops

    Dev->>Gradle: ./gradlew generateGroundTruth
    Gradle->>Docker: docker run skainet-ground-truth
    Docker->>PyTorch: Execute test cases

    loop For each @Executable function
        PyTorch->>PyTorch: Run operation (conv2d, relu, etc.)
        PyTorch->>GGUF: Serialize inputs + outputs
    end

    Docker-->>Gradle: Generation complete

    Dev->>Gradle: ./gradlew jvmTest
    Gradle->>Validator: Run validation tests

    loop For each GGUF file
        Validator->>GGUF: Load test case
        Validator->>SKaiNET: Execute same operation
        SKaiNET-->>Validator: Actual output
        Validator->>Validator: Compare with tolerance
    end

    Validator-->>Dev: Test results
```

## Component Architecture

### 1. Ground Truth Generation (Python)

```mermaid
flowchart LR
    subgraph "skainet-ground-truth/pytorch/src"
        TS1[TS-001/<br/>Convolution]
        TS2[TS-002/<br/>Slicing]
        TS3[TS-003/<br/>Flatten]
        TSN[TS-XXX/<br/>...]
    end

    subgraph "Test Suite Structure"
        UC1[UC-001.py]
        UC2[UC-002.py]
        UCN[UC-XXX.py]
    end

    subgraph "Test Function"
        EXEC["@Executable('description')<br/>def operation():<br/>  inputs = [...]<br/>  output = pytorch_op(...)<br/>  return inputs, output"]
    end

    TS1 --> UC1
    TS1 --> UC2
    UC1 --> EXEC
```

### 2. Execution & Serialization (gradienttracer)

```mermaid
flowchart TB
    subgraph "gt.exec (CLI)"
        CLI["gt src results"]
    end

    subgraph "gt.core"
        DISCOVER[iterate_and_execute<br/>Discover TS-XXX/UC-YYY.py]
        LOAD[load_module_from_file<br/>Dynamic import]
        FIND[find_executable_functions<br/>Find @Executable]
        EXECUTE[Execute function<br/>Get inputs & output]
    end

    subgraph "gt.pytorch.io.writer"
        CONVERT[Convert to F32<br/>Little Endian]
        WRITE[store_experiment_as_gguf<br/>Write GGUF file]
    end

    CLI --> DISCOVER
    DISCOVER --> LOAD
    LOAD --> FIND
    FIND --> EXECUTE
    EXECUTE --> CONVERT
    CONVERT --> WRITE
```

### 3. GGUF File Structure

```mermaid
graph TB
    subgraph "GGUF File Contents"
        META[Metadata]
        INPUTS[Input Tensors]
        OUTPUT[Result Tensor]
    end

    subgraph "Metadata Fields"
        DESC["general.description<br/>'Basic 2D Convolution'"]
        NAME["general.name<br/>'basic_2D_convolution'"]
    end

    subgraph "Tensor Data"
        IN0["input_0<br/>shape: [1,3,32,32]<br/>dtype: F32"]
        IN1["input_1 (weight)<br/>shape: [16,3,3,3]<br/>dtype: F32"]
        RES["result<br/>shape: [1,16,30,30]<br/>dtype: F32"]
    end

    META --> DESC
    META --> NAME
    INPUTS --> IN0
    INPUTS --> IN1
    OUTPUT --> RES
```

### 4. Validation (Kotlin)

```mermaid
flowchart TB
    subgraph "GroundTruthLoader"
        LOAD_GGUF[Load GGUF file]
        PARSE[Parse metadata & tensors]
        CREATE_TC[Create GroundTruthTestCase]
    end

    subgraph "OperationExecutor"
        MAP_OP[Map operation name<br/>to TensorOps method]
        CREATE_TENSOR[Create SKaiNET tensors<br/>from ground truth data]
        EXEC_OP[Execute operation]
    end

    subgraph "TensorAssertions"
        COMPARE[Compare output tensors]
        CHECK_SHAPE[Verify shapes match]
        CHECK_VALUES[Check values within tolerance]
    end

    subgraph "GroundTruthValidator"
        ORCHESTRATE[Orchestrate validation]
        REPORT[Generate report]
    end

    LOAD_GGUF --> PARSE
    PARSE --> CREATE_TC
    CREATE_TC --> MAP_OP
    MAP_OP --> CREATE_TENSOR
    CREATE_TENSOR --> EXEC_OP
    EXEC_OP --> COMPARE
    COMPARE --> CHECK_SHAPE
    CHECK_SHAPE --> CHECK_VALUES
    CHECK_VALUES --> REPORT
```

## Tolerance Configuration

Different operations require different tolerances due to floating-point precision:

```mermaid
graph LR
    subgraph "Tolerance Levels"
        STRICT["STRICT<br/>1e-6<br/>Basic arithmetic"]
        STANDARD["STANDARD<br/>1e-5<br/>Most operations"]
        RELAXED["RELAXED<br/>1e-4<br/>Transcendentals"]
        GRADIENT["GRADIENT<br/>1e-4<br/>Backprop"]
    end

    subgraph "Operations"
        ADD[add, subtract<br/>multiply, divide]
        CONV[conv2d, matmul]
        ACT[sigmoid, gelu<br/>softmax]
        GRAD[gradients]
    end

    ADD --> STRICT
    CONV --> STANDARD
    ACT --> RELAXED
    GRAD --> GRADIENT
```

## Test Suite Organization

```
skainet-ground-truth/pytorch/
├── Dockerfile              # Docker image definition
├── pyproject.toml          # Python dependencies
├── requirements.txt
├── src/
│   ├── TS-001/            # Convolution operations
│   │   ├── UC-001.py      # Basic conv2d
│   │   ├── UC-002.py      # Strided conv2d
│   │   └── ...
│   ├── TS-002/            # Slicing operations
│   ├── TS-003/            # Flatten operations
│   ├── TS-006/            # Broadcasting
│   └── TS-007/            # Advanced slicing
└── results/               # Generated GGUF files (not in git)
    ├── TS-001/
    │   ├── TS_001_UC_001.basic_2D_convolution.gguf
    │   └── ...
    └── ...
```

## Gradle Tasks

```mermaid
graph TB
    subgraph "Ground Truth Tasks"
        BUILD["buildGroundTruthDocker<br/>Build Docker image"]
        GENERATE["generateGroundTruth<br/>Run Docker, create GGUF"]
        CLEAN["cleanGroundTruth<br/>Delete GGUF files"]
        LIST["listGroundTruth<br/>Show available tests"]
    end

    subgraph "Test Tasks"
        TEST["jvmTest<br/>Run validation tests"]
    end

    BUILD --> GENERATE
    GENERATE --> TEST
```

### Usage

```bash
# First time setup: build Docker image
./gradlew :skainet-test:skainet-test-groundtruth:buildGroundTruthDocker

# Generate ground truth GGUF files
./gradlew :skainet-test:skainet-test-groundtruth:generateGroundTruth

# Run validation tests
./gradlew :skainet-test:skainet-test-groundtruth:jvmTest

# List available test suites
./gradlew :skainet-test:skainet-test-groundtruth:listGroundTruth

# Clean generated files
./gradlew :skainet-test:skainet-test-groundtruth:cleanGroundTruth
```

## Adding New Test Cases

### 1. Create Python Test (in skainet-ground-truth)

```python
# src/TS-001/UC-009.py
import torch
from gt.core import Executable

@Executable("Dilated Convolution with stride 2")
def dilated_conv_stride2():
    x = torch.randn(1, 3, 32, 32, requires_grad=True)
    conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, dilation=2)
    y = conv(x)
    return [x, conv.weight, conv.bias], y
```

### 2. Regenerate Ground Truth

```bash
./gradlew generateGroundTruth
```

### 3. Add Kotlin Validation (optional - for specific params)

```kotlin
@Test
fun `validate dilated conv with stride 2`() {
    val testCase = GroundTruthLoader.load("path/to/test.gguf")
    validator.assertValid(testCase, params = operationParams {
        stride(2)
        dilation(2)
    })
}
```

## Validation Report Format

```
============================================================
Validation Report: Basic 2D Convolution
============================================================
Operation: conv2d
Test Suite: TS-001
Use Case: UC-001
Source: results/TS-001/TS_001_UC_001.basic_2D_convolution.gguf

Forward Pass: PASSED
  Max absolute difference: 2.384185e-07
  Max relative difference: 1.192092e-06

Overall: PASSED
============================================================
```

## Error Handling

```mermaid
flowchart TB
    START[Start Validation] --> CHECK_AVAIL{Ground Truth<br/>Available?}

    CHECK_AVAIL -->|No| SKIP[Skip Test<br/>AssumptionViolatedException]
    CHECK_AVAIL -->|Yes| LOAD[Load GGUF]

    LOAD --> CHECK_OP{Operation<br/>Supported?}

    CHECK_OP -->|No| FAIL_OP[Fail: Unsupported<br/>Operation]
    CHECK_OP -->|Yes| EXECUTE[Execute in SKaiNET]

    EXECUTE --> COMPARE{Values<br/>Match?}

    COMPARE -->|Yes| PASS[PASS]
    COMPARE -->|No| FAIL_VAL[FAIL: Value Mismatch<br/>Show diff details]
```

## CI/CD Integration

```yaml
# .github/workflows/ground-truth.yml
name: Ground Truth Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker
        uses: docker/setup-buildx-action@v3

      - name: Build ground truth Docker image
        run: ./gradlew buildGroundTruthDocker

      - name: Generate ground truth
        run: ./gradlew generateGroundTruth

      - name: Run validation tests
        run: ./gradlew :skainet-test:skainet-test-groundtruth:jvmTest

      - name: Upload test reports
        uses: actions/upload-artifact@v4
        with:
          name: ground-truth-reports
          path: '**/build/reports/tests/**'
```

## Summary

The ground truth testing architecture provides:

1. **Isolation**: PyTorch runs in Docker, no Python setup needed in SKaiNET
2. **Reproducibility**: GGUF files capture exact tensor values
3. **Automation**: Gradle tasks handle Docker orchestration
4. **Flexibility**: Easy to add new test cases in Python
5. **Detailed Reporting**: Clear pass/fail with numerical differences
6. **CI-Ready**: Integrates with standard CI/CD pipelines
