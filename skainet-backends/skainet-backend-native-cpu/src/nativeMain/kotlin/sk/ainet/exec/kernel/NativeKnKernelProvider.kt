package sk.ainet.exec.kernel

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.usePinned
import sk.ainet.backend.api.kernel.Fp32MatmulKernel
import sk.ainet.backend.api.kernel.KernelProvider
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.backend.api.kernel.Q4KMatmulKernel
import sk.ainet.backend.api.kernel.Q4_0MatmulKernel
import sk.ainet.backend.api.kernel.Q5KMatmulKernel
import sk.ainet.backend.api.kernel.Q5_0MatmulKernel
import sk.ainet.backend.api.kernel.Q5_1MatmulKernel
import sk.ainet.backend.api.kernel.Q6KMatmulKernel
import sk.ainet.backend.api.kernel.Q8_0MatmulKernel
import sk.ainet.kernels.cinterop.skainet_q4_0_matmul
import sk.ainet.kernels.cinterop.skainet_q4k_matmul
import sk.ainet.kernels.cinterop.skainet_q5_0_matmul
import sk.ainet.kernels.cinterop.skainet_q5_1_matmul
import sk.ainet.kernels.cinterop.skainet_q6k_matmul
import sk.ainet.kernels.cinterop.skainet_q8_0_matmul

/**
 * Kotlin/Native [KernelProvider] backed by the hand-written C kernels via
 * cinterop (static archive `libskainet_kernels.a`) — the K/N analogue of the
 * JVM `NativeKernelProvider` (FFM). Priority 100, above the commonMain scalar
 * reference (0). On `linuxArm64` the linked archive carries the NEON paths.
 *
 * **Registration is manual on K/N** (no `ServiceLoader`): a consumer calls
 * [installNativeKernels] once at startup. [Q5KMatmulKernel] (the FunctionGemma
 * Q5_K_M hot path) plus Q4_K / Q6_K / Q8_0 / Q4_0 / Q5_0 / Q5_1 are wired; the
 * rest cascade to the scalar provider.
 */
@OptIn(ExperimentalForeignApi::class)
public object NativeKnKernelProvider : KernelProvider {
    override val name: String = "native-cinterop"
    override val priority: Int = 100

    // Statically linked — the symbols are always present once the binary links
    // libskainet_kernels.a, so the provider is unconditionally available.
    override fun isAvailable(): Boolean = true

    // Abstract on KernelProvider (no default) — no native FP32 SGEMM wrapper yet.
    override fun matmulFp32(): Fp32MatmulKernel? = null

    override fun matmulQ5K(): Q5KMatmulKernel = NativeKnQ5KMatmulKernel
    override fun matmulQ4K(): Q4KMatmulKernel = NativeKnQ4KMatmulKernel
    override fun matmulQ6K(): Q6KMatmulKernel = NativeKnQ6KMatmulKernel
    override fun matmulQ8_0(): Q8_0MatmulKernel = NativeKnQ8_0MatmulKernel
    override fun matmulQ4_0(): Q4_0MatmulKernel = NativeKnQ4_0MatmulKernel
    override fun matmulQ5_0(): Q5_0MatmulKernel = NativeKnQ5_0MatmulKernel
    override fun matmulQ5_1(): Q5_1MatmulKernel = NativeKnQ5_1MatmulKernel
}

/**
 * Register [NativeKnKernelProvider] in the process-wide [KernelRegistry]. Idempotent
 * (re-registering the same instance is a no-op). Call once at startup before any
 * `ops.matmul` on quantized weights.
 *
 * For quant types without a C kernel also register the commonMain
 * `ScalarKernelProvider` (from `skainet-backend-cpu`) as the fallback — it lives
 * in a different module, so the consumer wires it:
 * `KernelRegistry.register(ScalarKernelProvider)`.
 */
public fun installNativeKernels() {
    KernelRegistry.register(NativeKnKernelProvider)
}

@OptIn(ExperimentalForeignApi::class)
public object NativeKnQ4KMatmulKernel : Q4KMatmulKernel {
    private const val BLOCK_SIZE = 256
    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "NativeKnQ4KMatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0 || inputDim == 0) return
        input.usePinned { i -> weight.usePinned { w -> output.usePinned { o ->
            skainet_q4k_matmul(
                i.addressOf(0), inputOffset,
                w.addressOf(0).reinterpret(), weightByteOffset,
                inputDim, outputDim,
                o.addressOf(0), outputOffset,
            )
        } } }
    }
}

@OptIn(ExperimentalForeignApi::class)
public object NativeKnQ6KMatmulKernel : Q6KMatmulKernel {
    private const val BLOCK_SIZE = 256
    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "NativeKnQ6KMatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0 || inputDim == 0) return
        input.usePinned { i -> weight.usePinned { w -> output.usePinned { o ->
            skainet_q6k_matmul(
                i.addressOf(0), inputOffset,
                w.addressOf(0).reinterpret(), weightByteOffset,
                inputDim, outputDim,
                o.addressOf(0), outputOffset,
            )
        } } }
    }
}

@OptIn(ExperimentalForeignApi::class)
public object NativeKnQ8_0MatmulKernel : Q8_0MatmulKernel {
    private const val BLOCK_SIZE = 32
    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "NativeKnQ8_0MatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0 || inputDim == 0) return
        input.usePinned { i -> weight.usePinned { w -> output.usePinned { o ->
            skainet_q8_0_matmul(
                i.addressOf(0), inputOffset,
                w.addressOf(0).reinterpret(), weightByteOffset,
                inputDim, outputDim,
                o.addressOf(0), outputOffset,
            )
        } } }
    }
}

@OptIn(ExperimentalForeignApi::class)
public object NativeKnQ4_0MatmulKernel : Q4_0MatmulKernel {
    private const val BLOCK_SIZE = 32
    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "NativeKnQ4_0MatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0 || inputDim == 0) return
        input.usePinned { i -> weight.usePinned { w -> output.usePinned { o ->
            skainet_q4_0_matmul(
                i.addressOf(0), inputOffset,
                w.addressOf(0).reinterpret(), weightByteOffset,
                inputDim, outputDim,
                o.addressOf(0), outputOffset,
            )
        } } }
    }
}

@OptIn(ExperimentalForeignApi::class)
public object NativeKnQ5_0MatmulKernel : Q5_0MatmulKernel {
    private const val BLOCK_SIZE = 32
    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "NativeKnQ5_0MatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0 || inputDim == 0) return
        input.usePinned { i -> weight.usePinned { w -> output.usePinned { o ->
            skainet_q5_0_matmul(
                i.addressOf(0), inputOffset,
                w.addressOf(0).reinterpret(), weightByteOffset,
                inputDim, outputDim,
                o.addressOf(0), outputOffset,
            )
        } } }
    }
}

@OptIn(ExperimentalForeignApi::class)
public object NativeKnQ5_1MatmulKernel : Q5_1MatmulKernel {
    private const val BLOCK_SIZE = 32
    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "NativeKnQ5_1MatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0 || inputDim == 0) return
        input.usePinned { i -> weight.usePinned { w -> output.usePinned { o ->
            skainet_q5_1_matmul(
                i.addressOf(0), inputOffset,
                w.addressOf(0).reinterpret(), weightByteOffset,
                inputDim, outputDim,
                o.addressOf(0), outputOffset,
            )
        } } }
    }
}
