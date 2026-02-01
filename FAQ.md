# Frequently asked questions

## Q:How project compare with existing or historical efforts.

**A**: SKaiNET differs from most existing machine-learning frameworks in its end-to-end, **on-device-first** design using Kotlin. Established frameworks such as PyTorch and TensorFlow are Python-based tools focused to run on machines with powerful HW(optimized for mostly cloud environments). While they can be adapted for mobile or embedded use, this typically requires additional non-trivial effort, like model conversion pipelines, or vendor-specific runtimes, which makes an adoption more difficult and brings performance and usability trade-offs from their original design goals.

Native on-device support is mostly driven by chip producers companies providing isolated solutions for their own chips, or it is often tightly coupled to their own platforms—for example, Apple’s frameworks optimized for Apple Silicon or Google’s solutions focused on Android.

By starting from scratch, SKaiNET was able to make architectural decisions that are difficult for mature frameworks due to legacy constraints and backward compatibility. Using Kotlin and Kotlin Multiplatform enables a single, type-safe codebase, structured concurrency, and direct compilation to native targets, reducing platform-specific complexity. SKaiNET utilizing its own compiler steps and transformations and building up on existing powerful compiler infrastructure (MLIR, LLVM) can bring native performance with improved developer experience.
As a result, SKaiNET complements rather than competes with existing frameworks by addressing a distinct gap: open, cross-platform, native, on-device AI without dependence on cloud-centric tooling or vendor-locked runtimes.
