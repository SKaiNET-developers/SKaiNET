# Technology Stack & Build System

## Build System

- **Gradle**: Kotlin DSL with version catalogs (`gradle/libs.versions.toml`)
- **JVM Toolchain**: Java 21 (enforced across all modules)
- **Android**: minSdk 24, compileSdk 36, target JVM 11 for compatibility
- **Shared Build Logic**: Located in `build-logic/` for consistent configuration

## Core Technologies

- **Kotlin Multiplatform**: 2.2.21
- **Platforms**: JVM, Android, iOS (Arm64/Simulator), macOS Arm64, Linux (x64/Arm64), JS, WASM
- **Serialization**: kotlinx-serialization-json 1.9.0
- **Coroutines**: kotlinx-coroutines 1.10.2
- **I/O**: kotlinx-io 0.8.2
- **Networking**: Ktor client 3.3.3

## Code Generation & Processing

- **KSP**: Kotlin Symbol Processing 2.2.21-2.0.4 for operator metadata generation
- **KotlinPoet**: 2.2.0 for code generation
- **Protobuf**: pbandk 0.16.0 for ONNX model parsing

## Testing & Quality

- **Testing**: kotlin-test with kotlinx-coroutines-test
- **Coverage**: Kover 0.9.3 with HTML/XML reports
- **API Compatibility**: binary-compatibility-validator 0.18.1
- **Documentation**: Dokka 2.1.0 + AsciiDoc

## Key Build Commands

### Development Workflow
```bash
# Full CI-equivalent build and test
./gradlew clean assemble allTests

# Quick module-specific testing
./gradlew :module:allTests

# Generate documentation
./gradlew generateOperatorDocs validateOperatorSchema

# Coverage reports
./gradlew koverHtmlReport

# API compatibility check
./gradlew apiCheck
```

### JVM-Specific Configuration
- **Vector API**: Enabled with `--enable-preview --add-modules jdk.incubator.vector`
- **SIMD Optimization**: Available on JVM backend for performance-critical operations

### Android Configuration
- Compatibility target: JVM 11 bytecode
- Namespace pattern: `sk.ainet.{module}`
- Uses AndroidX libraries

### Publishing
- **Maven Central**: Automated publishing with Vanniktech plugin
- **Group ID**: `sk.ainet.core`
- **Version**: 0.5.0 (semantic versioning)
- **Signing**: Configurable for releases

## Module Dependencies Pattern

- Core modules use `explicitApi()` mode
- KSP processors generate operator metadata
- Backend modules depend on lang-core
- I/O modules are optional and modular
- Test dependencies include model fixtures