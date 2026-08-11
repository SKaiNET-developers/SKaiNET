package sk.ainet.exec.kernel

import sk.ainet.backend.api.kernel.Bf16MatmulKernel
import sk.ainet.backend.api.kernel.Fp16MatmulKernel
import sk.ainet.backend.api.kernel.Fp32MatmulKernel
import sk.ainet.backend.api.kernel.KernelProvider
import sk.ainet.backend.api.kernel.MemSegKernelProvider
import sk.ainet.backend.api.kernel.Q4KMatmulKernel
import sk.ainet.backend.api.kernel.Q4KMemSegMatmulKernel
import sk.ainet.backend.api.kernel.Q4_0MatmulKernel
import sk.ainet.backend.api.kernel.Q5KMatmulKernel
import sk.ainet.backend.api.kernel.Q5_0MatmulKernel
import sk.ainet.backend.api.kernel.Q5_1MatmulKernel
import sk.ainet.backend.api.kernel.Q6KMatmulKernel
import sk.ainet.backend.api.kernel.Q8_0MatmulKernel

/**
 * Native (FFM) [KernelProvider] / [MemSegKernelProvider]. Sits at
 * priority `100`, above [PanamaVectorKernelProvider] (`50`) and the
 * scalar reference (`0`).
 *
 * Availability is gated on [NativeQ4KMatmulKernel.isAvailable] — the
 * bundled `libskainet_kernels` shared library has to load AND
 * `skainet_q4k_matmul` has to resolve via FFM. When either fails
 * (missing arch, sandbox, JDK without FFM, kill-switch),
 * `KernelRegistry.bestAvailable()` cleanly cascades to
 * [PanamaVectorKernelProvider] at priority 50.
 *
 * The MemSeg surface ([matmulQ4KMemSeg]) is the JVM-only zero-copy
 * path for mmap'd Q4_K weights — sized for inference loops that
 * project against pre-loaded `MemorySegment`-backed tensors. Heap
 * callers stick with [matmulQ4K]; both wrap the same C symbol so
 * outputs are bit-for-bit identical.
 *
 * ## Consuming this module
 *
 * The whole `skainet-backend-native-cpu` module is **JVM-only**: it
 * depends on `java.lang.foreign.*`, which exists on Java SE only —
 * not on Kotlin/Native, JS, Wasm, or Android Runtime. The module
 * declares only a `jvm()` Kotlin target; no klib variants are
 * published.
 *
 * KMP consumers MUST add the dependency to `jvmMain` only, never to
 * `commonMain`:
 *
 * ```kotlin
 * sourceSets {
 *     val jvmMain by getting {
 *         dependencies {
 *             implementation("sk.ainet.core:skainet-backend-native-cpu:<version>")
 *         }
 *     }
 * }
 * ```
 *
 * Putting it in `commonMain.dependencies` causes Gradle to fail
 * resolution on every non-JVM target with a long string of "Couldn't
 * resolve dependency 'sk.ainet.core:skainet-backend-native-cpu' in
 * 'commonMain' for all target platforms" warnings. JVM-only consumers
 * (plain `kotlin("jvm")` modules) can use the regular `dependencies`
 * block.
 *
 * Auto-discovery: `KernelServiceLoader.installAll()` scans
 * `META-INF/services/sk.ainet.backend.api.kernel.KernelProvider` on
 * the classpath and registers everything it finds. With the JAR on
 * the classpath there is nothing else to wire — no manual
 * `KernelRegistry.register()` call.
 *
 * Shadow-jar consumers: `mergeServiceFiles()` on shadow plugin
 * 9.4.x has a known bug that silently drops one of the two co-
 * located service files when both `skainet-backend-cpu` and
 * `skainet-backend-native-cpu` are on the classpath — see the
 * `kllama-cli` build script in `SKaiNET-transformers` for a working
 * `doLast` workaround that rebuilds the union.
 *
 * Staged rollout cursor (see `native-ffm-plan` asciidoc):
 *  - PR 2: real Q4_K matmul wired into the heap SPI.
 *  - PR 3: MemSeg-input zero-copy sibling.
 *  - PR 5: native FP32 matmul wired into [matmulFp32].
 *  - Now: native `matmulQ5K`, `matmulQ6K`, `matmulQ8_0`, `matmulQ4_0` all wired.
 *  - Now: native `matmulFp16`, closing the gap against `matmulBf16` (#887).
 *    Every narrow-float accessor the SPI declares is wired here; a format
 *    served natively on one side and by the JVM fallback on the other looks
 *    like a slow kernel rather than a missing one.
 */
public object NativeKernelProvider : KernelProvider, MemSegKernelProvider {
    override val name: String = "native-ffm"
    override val priority: Int = 100

    override fun isAvailable(): Boolean = NativeQ4KMatmulKernel.isAvailable()

    override fun matmulFp32(): Fp32MatmulKernel? =
        if (NativeFp32MatmulKernel.isAvailable()) NativeFp32MatmulKernel else null

    override fun matmulQ4K(): Q4KMatmulKernel? =
        if (NativeQ4KMatmulKernel.isAvailable()) NativeQ4KMatmulKernel else null

    override fun matmulQ4KMemSeg(): Q4KMemSegMatmulKernel? =
        if (NativeQ4KMemSegMatmulKernel.isAvailable()) NativeQ4KMemSegMatmulKernel else null

    override fun matmulBf16(): Bf16MatmulKernel? =
        if (NativeBf16MatmulKernel.isAvailable()) NativeBf16MatmulKernel else null

    override fun matmulFp16(): Fp16MatmulKernel? =
        if (NativeFp16MatmulKernel.isAvailable()) NativeFp16MatmulKernel else null

    override fun matmulQ8_0(): Q8_0MatmulKernel? =
        if (NativeQ8_0MatmulKernel.isAvailable()) NativeQ8_0MatmulKernel else null

    override fun matmulQ4_0(): Q4_0MatmulKernel? =
        if (NativeQ4_0MatmulKernel.isAvailable()) NativeQ4_0MatmulKernel else null

    override fun matmulQ5K(): Q5KMatmulKernel? =
        if (NativeQ5KMatmulKernel.isAvailable()) NativeQ5KMatmulKernel else null

    override fun matmulQ6K(): Q6KMatmulKernel? =
        if (NativeQ6KMatmulKernel.isAvailable()) NativeQ6KMatmulKernel else null

    override fun matmulQ5_0(): Q5_0MatmulKernel? =
        if (NativeQ5_0MatmulKernel.isAvailable()) NativeQ5_0MatmulKernel else null

    override fun matmulQ5_1(): Q5_1MatmulKernel? =
        if (NativeQ5_1MatmulKernel.isAvailable()) NativeQ5_1MatmulKernel else null
}
