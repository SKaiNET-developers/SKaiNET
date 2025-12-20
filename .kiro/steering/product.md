# SKaiNET Product Overview

SKaiNET is an open-source deep learning framework written in Kotlin Multiplatform, designed to enable modern AI-powered applications with ease. It provides a developer-friendly DSL for neural networks and supports multiple execution backends.

## Key Features

- **Kotlin Multiplatform**: Runs on JVM, Android, iOS, JS, WASM, and Native platforms
- **Neural Network DSL**: Intuitive Kotlin DSL for model definition with type safety
- **Hardware Compilation**: StableHLO/MLIR compilation to any XLA-supported hardware (CPU, GPU, TPU)
- **Model I/O**: Support for ONNX, GGUF, and JSON model formats
- **Experimental KAN**: Kolmogorov-Arnold Networks implementation
- **Data Loading**: Built-in loaders for common datasets (MNIST, etc.)
- **Training/Evaluation**: Easy phase switching with execution contexts

## Current Version

Version 0.5.0 (as of December 2025) includes ONNX import, CLI tooling, YOLOv8 model support, and image I/O capabilities.

## Target Use Cases

- Kotlin applications requiring ML inference
- Cross-platform AI applications
- Research and experimentation with neural networks
- Educational projects and Kotlin Notebooks
- CLI tools for model processing and conversion

## Architecture Philosophy

- DSL-driven development with compile-time safety
- Backend abstraction separating high-level operations from execution
- Multiplatform-first design
- Explicit API surface with comprehensive documentation