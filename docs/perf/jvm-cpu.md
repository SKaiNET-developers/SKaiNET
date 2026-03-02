### JVM CPU Backend Performance Benchmarks (JMH)

This page explains how to run the JMH benchmarks for the JVM CPU backend and how to capture evidence for performance targets.

#### What’s included
- Elementwise: FP32 `add` on 1,000,000 elements
- Reductions: FP32 `sum` and `mean` on 1,000,000 elements
- Matmul: FP32 square `matmul` with sizes 256, 512, and 1024

Benchmarks are implemented in module:
- `:skainet-backends:benchmarks:jvm-cpu-jmh`

Source files:
- `src/jmh/kotlin/sk/ainet/bench/ElementwiseAdd1MBench.kt`
- `src/jmh/kotlin/sk/ainet/bench/Reductions1MBench.kt`
- `src/jmh/kotlin/sk/ainet/bench/MatmulBench.kt`

#### Prerequisites
- JDK 21+ (JDK 22 toolchain configured by Gradle)
- Gradle will pass required JVM flags:
  - `--enable-preview`
  - `--add-modules jdk.incubator.vector`

For Java 25-specific performance advantages, see [Java 25 CPU Backend](java-25-cpu-backend.md).

#### Feature flags
You can toggle acceleration paths at runtime using system properties or environment variables:
- Vector acceleration:
  - `-Dskainet.cpu.vector.enabled=true|false`
  - or `SKAINET_CPU_VECTOR_ENABLED=true|false`
- BLAS via Panama (matmul heuristic for larger sizes):
  - `-Dskainet.cpu.blas.enabled=true|false`
  - or `SKAINET_CPU_BLAS_ENABLED=true|false`

Each benchmark also exposes `@Param` to toggle these flags without modifying Gradle args.

#### How to run all benchmarks
From repository root:

```
./gradlew :skainet-backends:benchmarks:jvm-cpu-jmh:jmh
```

This will build and execute all JMH benchmarks with the default parameters defined in sources.

#### Run specific benchmarks
- Elementwise add (both vector on/off):
```
./gradlew :skainet-backends:benchmarks:jvm-cpu-jmh:jmh \
  -Pjmh.include=ElementwiseAdd1MBench
```

- Reductions (vector on/off):
```
./gradlew :skainet-backends:benchmarks:jvm-cpu-jmh:jmh \
  -Pjmh.include=Reductions1MBench
```

- Matmul, all sizes, with vector on and BLAS on:
```
./gradlew :skainet-backends:benchmarks:jvm-cpu-jmh:jmh \
  -Pjmh.include=MatmulBench \
  -Pjmh.param.vectorEnabled=true \
  -Pjmh.param.blasEnabled=true
```

- Matmul at 512 only, comparing BLAS on/off with vector on:
```
./gradlew :skainet-backends:benchmarks:jvm-cpu-jmh:jmh \
  -Pjmh.include=MatmulBench \
  -Pjmh.param.size=512 \
  -Pjmh.param.vectorEnabled=true \
  -Pjmh.param.blasEnabled=true,false
```

Notes:
- You can also pass system properties via `-D` if preferred (e.g., `-Dskainet.cpu.vector.enabled=false`).
- JMH JSON/text results can be configured via standard JMH plugin options if you need files for CI artifacts.

#### Recording environment details
Include at minimum:
- CPU model, cores/threads, base/boost clock
- RAM size and speed
- OS version
- JDK version and vendor
- Gradle version
- JVM flags in use (`--enable-preview --add-modules jdk.incubator.vector`)
- SKaiNET flags used (vector, BLAS)

#### Performance targets (to be validated on your hardware)
- ≥ 4× speedup on FP32 `matmul` 512×512 vs baseline scalar
- ≥ 3× speedup on FP32 `add` with 1M elements vs baseline scalar

Use the above commands to produce “vector=false/blas=false” baselines vs “vector=true[/blas=true]” accelerated runs. Capture best-of or median-of JMH results as evidence and include raw tables in this document when available.
