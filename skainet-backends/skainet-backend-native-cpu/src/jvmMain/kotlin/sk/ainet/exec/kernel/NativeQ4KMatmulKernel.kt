package sk.ainet.exec.kernel

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle
import sk.ainet.backend.api.kernel.Q4KMatmulKernel

/**
 * Native (FFM) implementation of [Q4KMatmulKernel].
 *
 * Wraps the bundled C symbol
 *
 *   void skainet_q4k_matmul(
 *       const float* input, int32_t input_offset,
 *       const uint8_t* weight, int32_t weight_byte_offset,
 *       int32_t input_dim, int32_t output_dim,
 *       float* output, int32_t output_offset);
 *
 * The C kernel implements the same lazy-`dmin` accumulation as
 * [PanamaVectorQ4KMatmulKernel] (sum input·code and sum input per
 * sub-block, combine via `d * scaleIdx[s] * codeSum - dMin * minIdx[s] * inputSum`)
 * and shares the canonical 256-element / 144-byte super-block layout.
 *
 * Numerical parity vs the Panama kernel is asserted by
 * [NativeQ4KMatmulKernelParityTest] within `1e-4` relative tolerance,
 * matching the parity bar `PanamaVectorQ4KMatmulKernelTest` uses.
 *
 * PR 2 of the staged native-FFM rollout: ships a single-threaded
 * scalar C kernel (`-O3 -ffast-math`, auto-vectorized inner loop).
 * NEON / AVX2 intrinsics, `MemorySegment`-input zero-copy variant,
 * and cross-arch CI shipping are deferred to PRs 3–5.
 */
internal object NativeQ4KMatmulKernel : Q4KMatmulKernel {

    private const val BLOCK_SIZE = 256

    fun isAvailable(): Boolean = handle != null

    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "NativeQ4KMatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0 || inputDim == 0) return
        val mh = handle
            ?: error("NativeQ4KMatmulKernel.matmul invoked while native library unavailable")

        // The native kernel writes outputDim floats and only reads
        // inputDim floats + (inputDim/256)*outputDim*144 weight bytes,
        // so the segments size exactly to those windows. Heap-array
        // segments would also work but allocating off-heap copies keeps
        // the native side oblivious to the JVM heap layout (and lets
        // the same wrapper take MemorySegment-backed inputs in PR 3).
        Arena.ofConfined().use { arena ->
            val inSeg = arena.allocate(
                inputDim.toLong() * java.lang.Float.BYTES,
                ValueLayout.JAVA_FLOAT.byteAlignment(),
            )
            val outSeg = arena.allocate(
                outputDim.toLong() * java.lang.Float.BYTES,
                ValueLayout.JAVA_FLOAT.byteAlignment(),
            )
            val weightBytesUsed = ((inputDim / BLOCK_SIZE).toLong() * outputDim) * 144L
            val weightSeg = arena.allocate(weightBytesUsed, 1L)

            MemorySegment.copy(input, inputOffset, inSeg, ValueLayout.JAVA_FLOAT, 0L, inputDim)
            MemorySegment.copy(weight, weightByteOffset, weightSeg, ValueLayout.JAVA_BYTE, 0L, weightBytesUsed.toInt())

            mh.invoke(
                inSeg, 0,
                weightSeg, 0,
                inputDim, outputDim,
                outSeg, 0,
            )

            MemorySegment.copy(outSeg, ValueLayout.JAVA_FLOAT, 0L, output, outputOffset, outputDim)
        }
    }

    private val handle: MethodHandle? by lazy {
        val lookup = NativeLibraryLoader.lookup() ?: return@lazy null
        val symbol = lookup.find("skainet_q4k_matmul").orElse(null) ?: return@lazy null
        val descriptor = FunctionDescriptor.ofVoid(
            ValueLayout.ADDRESS,    // input
            ValueLayout.JAVA_INT,   // input_offset
            ValueLayout.ADDRESS,    // weight
            ValueLayout.JAVA_INT,   // weight_byte_offset
            ValueLayout.JAVA_INT,   // input_dim
            ValueLayout.JAVA_INT,   // output_dim
            ValueLayout.ADDRESS,    // output
            ValueLayout.JAVA_INT,   // output_offset
        )
        runCatching { Linker.nativeLinker().downcallHandle(symbol, descriptor) }.getOrNull()
    }
}
