# Frequently asked questions

### Q: Why build this in Kotlin instead of using Python/PyTorch?

**A**:  Most real products aren’t deployed from notebooks — Kotlin teams need a Kotlin-native path to production that minimizes glue code and cross-language complexity. This isn't about replacing PyTorch. It's about closing the gap between Kotlin and production.

Python is great for research; Kotlin shines in production app stacks. The focus here is *deployment and integration pain*, not “beating PyTorch”.


---

### Q: Why Kotlin for ML at all??

**A**: Because Kotlin teams want type-safety, maintainability, and shared code across platforms—especially when ML is only one part of a larger product.

Kotlin DSLs make pipelines and inference code more readable and less error-prone. KMP enables sharing logic across JVM, Android, iOS, JS and WASM.

If Kotlin isn’t your primary stack, this probably isn't for you - and that’s okay.

---


### Q) How does the project compare with the existing or historical efforts?

SKaiNET differs from most existing machine-learning frameworks in its end-to-end, **on-device-first** design using Kotlin. Established frameworks such as PyTorch and TensorFlow are Python-based tools focused to run on machines with powerful HW(optimized for mostly cloud environments). While they can be adapted for mobile or embedded use, this typically requires additional non-trivial effort, like model conversion pipelines, or vendor-specific runtimes, which makes an adoption more difficult and brings performance and usability trade-offs from their original design goals.

Native on-device support is mostly driven by chip producers companies providing isolated solutions for their own chips, or it is often tightly coupled to their own platforms—for example, Apple’s frameworks optimized for Apple Silicon or Google’s solutions focused on Android.

By starting from scratch, SKaiNET was able to make architectural decisions that are difficult for mature frameworks due to legacy constraints and backward compatibility. Using Kotlin and Kotlin Multiplatform enables a single, type-safe codebase, structured concurrency, and direct compilation to native targets, reducing platform-specific complexity. SKaiNET utilizing its own compiler steps and transformations and building up on existing powerful compiler infrastructure (MLIR, LLVM) can bring native performance with improved developer experience.
As a result, SKaiNET complements rather than competes with existing frameworks by addressing a distinct gap: open, cross-platform, native, on-device AI without dependence on cloud-centric tooling or vendor-locked runtimes.
