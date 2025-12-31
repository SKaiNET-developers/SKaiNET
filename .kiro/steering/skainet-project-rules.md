# SKaiNET Project Development Rules

## Project Overview
SKaiNET is a Kotlin Multiplatform deep learning framework designed for modern AI applications. It features a DSL for neural networks, multiple backends (CPU, with GPU planned), and comprehensive I/O support for various model formats including ONNX and GGUF.

## Architecture & Module Organization

### Core Module Structure
- **skainet-lang/**: Core DSL, operator metadata, and KSP processor
  - `skainet-lang-core`: Main DSL and tensor operations
  - `skainet-lang-models`: Pre-built model definitions
  - `skainet-lang-ksp-*`: Annotation processing for operator generation
  - `skainet-kan`: Kolmogorov-Arnold Network implementation
- **skainet-backends/**: Runtime execution engines
  - `skainet-backend-cpu`: CPU implementation with SIMD optimization
- **skainet-compile/**: Graph transformation and compilation
  - `skainet-compile-core`: Core compilation infrastructure
  - `skainet-compile-dag`: Directed acyclic graph operations
  - `skainet-compile-json`: JSON export/import
  - `skainet-compile-hlo`: High-level optimization
- **skainet-data/**: Data loading and preprocessing
- **skainet-io/**: Model format I/O (ONNX, GGUF, images)
- **skainet-models/**: Specific model implementations (YOLO, etc.)
- **skainet-apps/**: CLI tools and applications

### Key Design Principles
- Multiplatform-first: Support JVM, Android, iOS, JS, WASM, and Native
- DSL-driven: Provide intuitive Kotlin DSL for model definition
- Backend abstraction: Separate high-level operations from execution
- Explicit API: Use `explicitApi()` mode for public modules
- Type safety: Leverage Kotlin's type system for tensor operations

## Development Guidelines

### Build System & Dependencies
- Use Gradle with Kotlin DSL and version catalogs (`gradle/libs.versions.toml`)
- JVM toolchain: Java 21 (enforced across all modules)
- Android: minSdk 24, compileSdk 36, target JVM 11 for compatibility
- Follow semantic versioning (currently 0.5.0)
- Use shared build logic in `build-logic/` for consistent configuration

### Code Style & Conventions
- **Formatting**: 4-space indentation, 120-character line limit
- **Naming**: 
  - Types: `UpperCamelCase`
  - Functions/properties: `lowerCamelCase`
  - Test fixtures: `snake_case`
  - Backend kernels: suffix with `*Kernel`
- **Visibility**: Use explicit visibility modifiers (required by `explicitApi()`)
- **Documentation**: KDoc for all public APIs, especially DSL entry points
- **Experimental features**: Gate with `@OptIn` annotations

### Testing Strategy
- **Location**: Tests in `src/commonTest/` for shared logic, `src/jvmTest/` for platform-specific
- **Naming**: Test files as `FeatureNameTest.kt`, methods as `operation_expectedBehavior`
- **Frameworks**: Use `kotlin-test` and `kotlinx-coroutines-test`
- **Fixtures**: Share tensor test data through helper functions, avoid inline literals
- **Coverage**: Run `./gradlew koverHtmlReport` for coverage analysis
- **CI Validation**: Always run `./gradlew clean assemble allTests` before PRs

### Module Development Patterns
- **DSL Design**: Follow builder pattern with type-safe configuration
- **Backend Implementation**: Separate interface definition from platform-specific kernels
- **Operator Registration**: Use KSP for automatic operator metadata generation
- **Error Handling**: Provide clear error messages with context
- **Performance**: Consider SIMD optimization for JVM, platform-specific optimizations

## Specific Technology Guidelines

### Kotlin Multiplatform
- Use `expect`/`actual` for platform-specific implementations
- Prefer common implementations when possible
- Test on multiple platforms, especially for mathematical operations
- Handle platform differences in I/O and memory management

### Neural Network DSL
- Follow functional composition patterns
- Support both eager and graph execution modes
- Provide clear separation between model definition and execution
- Enable training/evaluation phase switching via `ExecutionContext`

### Tensor Operations
- Implement shape inference and broadcasting rules
- Support multiple data types (Float32, Int32, etc.)
- Provide both low-level and high-level APIs
- Ensure memory efficiency and proper cleanup

### Model I/O
- Support standard formats: ONNX, GGUF, JSON
- Implement robust parsing with clear error messages
- Handle version compatibility and format variations
- Provide conversion utilities between formats

## Git Workflow (GitFlow)

### Branch Strategy
- **main**: Production-ready releases only
- **develop**: Integration branch for features
- **feature/**: New features (`feature/transformer-attention`)
- **release/**: Release preparation (`release/1.0.0`)
- **hotfix/**: Critical production fixes (`hotfix/1.0.1`)

### Commit Guidelines
- Use imperative mood: "Add feature" not "Added feature"
- Reference issues: "Fix gradient clipping (closes #123)"
- Keep first line under 50 characters
- Include detailed description for complex changes

### Pull Request Process
- Always create PRs for merges to main/develop
- Require code review and passing tests
- Use `--no-ff` to preserve branch history
- Include comprehensive change descriptions

## Documentation & API Design

### Public API Requirements
- All public APIs must have KDoc documentation
- Include usage examples for complex features
- Document parameter constraints and return value meanings
- Specify thread safety and platform compatibility

### Generated Documentation
- Operator docs auto-generated via KSP in `operators.json`
- Run `./gradlew generateOperatorDocs` to update
- AsciiDoc files in `docs/` for comprehensive guides
- Dokka for API reference documentation

### Schema Validation
- Validate generated JSON against schemas
- Run `./gradlew validateOperatorSchema` before releases
- Maintain backward compatibility in public APIs

## Performance & Optimization

### Backend Performance
- Implement SIMD operations for JVM using Vector API
- Use platform-specific optimizations (Metal for iOS, etc.)
- Profile critical paths with JMH benchmarks
- Consider memory layout and cache efficiency

### Model Execution
- Support both eager and graph execution modes
- Implement operator fusion where beneficial
- Provide memory pooling for tensor allocation
- Enable mixed precision computation

## Security & Best Practices

### Dependency Management
- Keep dependencies up to date via version catalog
- Avoid transitive dependency conflicts
- Use minimal dependency sets per module
- Regular security audits of dependencies

### Data Handling
- Validate input data shapes and types
- Handle large models efficiently (streaming, chunking)
- Implement proper resource cleanup
- Consider memory constraints on mobile platforms

## Release Process

### Version Management
- Follow semantic versioning strictly
- Update CHANGELOG.md with detailed release notes
- Tag releases with version numbers
- Coordinate multiplatform artifact publishing

### Quality Gates
- All tests must pass on all platforms
- API compatibility validation via binary-compatibility-validator
- Documentation generation and validation
- Performance regression testing

## Common Patterns & Anti-Patterns

### Recommended Patterns
- Use DSL builders for complex configuration
- Implement proper resource management with `use` blocks
- Leverage Kotlin's type system for compile-time safety
- Provide both high-level convenience and low-level control APIs

### Anti-Patterns to Avoid
- Direct platform-specific code in common modules
- Mutable global state
- Blocking operations in suspend functions
- Memory leaks in tensor operations
- Inconsistent error handling across modules

## Integration Guidelines

### Adding New Operators
1. Define operator interface in `skainet-lang-core`
2. Add KSP annotations for metadata generation
3. Implement backend kernels in `skainet-backend-cpu`
4. Add DSL support and tests
5. Update documentation and examples

### Adding New Backends
1. Implement backend interface from `skainet-backends-core`
2. Provide platform-specific optimizations
3. Ensure compatibility with existing operators
4. Add comprehensive test coverage
5. Document performance characteristics

### Model Format Support
1. Add parser in appropriate `skainet-io` module
2. Implement conversion to internal representation
3. Handle format-specific features and limitations
4. Provide validation and error reporting
5. Add CLI tools if beneficial

This comprehensive rule set ensures consistent development practices across the SKaiNET project while maintaining the framework's multiplatform nature and performance goals.