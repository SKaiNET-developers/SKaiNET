# Project Structure & Organization

## Root Level Structure

```
SKaiNET/
├── build-logic/           # Shared Gradle build configuration
├── docs/                  # Documentation (AsciiDoc, examples, theory)
├── gradle/               # Gradle wrapper and version catalogs
├── skainet-lang/         # Core DSL and language features
├── skainet-backends/     # Execution backends
├── skainet-compile/      # Graph compilation and optimization
├── skainet-data/         # Data loading and preprocessing
├── skainet-io/           # Model format I/O
├── skainet-models/       # Specific model implementations
├── skainet-apps/         # CLI applications and tools
└── tools/                # Development and documentation tools
```

## Module Organization Patterns

### Language Modules (`skainet-lang/`)
- **skainet-lang-core**: Main DSL, tensor operations, execution contexts
- **skainet-lang-models**: Pre-built model definitions and utilities
- **skainet-lang-ksp-annotations**: Annotations for code generation
- **skainet-lang-ksp-processor**: KSP processor for operator metadata
- **skainet-kan**: Kolmogorov-Arnold Network implementation

### Backend Modules (`skainet-backends/`)
- **skainet-backend-cpu**: CPU implementation with SIMD optimization
- **benchmarks/jvm-cpu-jmh**: JMH performance benchmarks

### Compilation Modules (`skainet-compile/`)
- **skainet-compile-core**: Core compilation infrastructure
- **skainet-compile-dag**: Directed acyclic graph operations
- **skainet-compile-json**: JSON export/import functionality
- **skainet-compile-hlo**: High-level optimization passes

### I/O Modules (`skainet-io/`)
- **skainet-io-core**: Base I/O abstractions
- **skainet-io-gguf**: GGUF format support
- **skainet-io-image**: Image processing and conversion
- **skainet-io-onnx**: ONNX model parsing and import

### Data Modules (`skainet-data/`)
- **skainet-data-api**: Data loading abstractions
- **skainet-data-simple**: Simple dataset implementations (MNIST, etc.)

## Source Set Organization

Each multiplatform module follows this structure:
```
src/
├── commonMain/           # Shared implementation
├── commonTest/           # Shared tests
├── jvmMain/             # JVM-specific code
├── jvmTest/             # JVM-specific tests
├── androidMain/         # Android-specific code
├── nativeMain/          # Native shared code
├── iosMain/             # iOS-specific code
├── jsMain/              # JavaScript-specific code
└── wasmJsMain/          # WASM-specific code
```

## Key Architectural Principles

### Module Dependencies
- **Core Dependencies**: `skainet-lang-core` is the foundation
- **Backend Abstraction**: Backends depend on lang-core, not vice versa
- **Optional I/O**: I/O modules are independent and composable
- **Layered Architecture**: Clear separation between DSL, compilation, and execution

### Code Generation
- **KSP Integration**: Operator metadata generated at build time
- **Generated Resources**: Operators.json in `build/generated/ksp/metadata/commonMain/resources/`
- **Documentation**: Auto-generated docs in `docs/modules/operators/_generated_/`

### Platform-Specific Code
- **Expect/Actual**: Used for platform-specific implementations
- **Native Hierarchies**: Shared native code with platform-specific overrides
- **Performance Optimization**: Platform-specific SIMD and vector operations

### Testing Structure
- **Test Location**: `src/commonTest/` for shared logic, `src/jvmTest/` for platform-specific
- **Naming Convention**: `FeatureNameTest.kt` files, `operation_expectedBehavior` methods
- **Fixtures**: Shared test data through helper functions, avoid inline literals

### Build Configuration
- **Explicit API**: Public modules use `explicitApi()` mode
- **Toolchain**: JVM 21 enforced across all modules
- **Android Compatibility**: JVM 11 bytecode target for Android modules
- **Version Catalogs**: Centralized dependency management in `gradle/libs.versions.toml`

## Documentation Organization

```
docs/
├── examples/             # Usage examples and tutorials
├── theory/              # Mathematical and theoretical documentation
├── perf/                # Performance analysis and benchmarks
└── modules/operators/_generated_/  # Auto-generated operator docs
```

## Generated Artifacts

- **Operator Metadata**: JSON files with operator definitions
- **API Documentation**: Dokka-generated API docs
- **Coverage Reports**: Kover HTML/XML reports in `build/reports/`
- **Binary Compatibility**: API dump files for compatibility validation