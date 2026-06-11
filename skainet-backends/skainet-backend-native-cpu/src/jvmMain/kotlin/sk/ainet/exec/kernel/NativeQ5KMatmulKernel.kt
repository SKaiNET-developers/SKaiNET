package sk.ainet.exec.kernel

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle
import sk.ainet.backend.api.kernel.Q5KMatmulKernel

/**
 * Native (FFM) implementation of [Q5KMatmulKernel].
 *
 * Wraps the bundled C symbol
 *
 *   void skainet_q5k_matmul(
 *       const float* input, int32_t input_offset,
 *       const uint8_t* weight, int32_t weight_byte_offset,
 *       int32_t input_dim, int32_t output_dim,
 *       float* output, int32_t output_offset);
 *
 * Same lazy-`dmin` accumulation as [PanamaVectorQ5_KMatmulKernel] over
 * the canonical 256-element / 176-byte Q5_K super-block (the 5th bit of
 * each code comes from the `qh` plane). Numerical parity vs the Panama
 * kernel is asserted by [NativeQ5KMatmulKernelParityTest].
 *
 * Single-threaded scalar C (`-O3 -ffast-math`, auto-vectorized inner
 * loop); a hand-written NEON path is layered on behind `__ARM_NEON`.
 */
internal object NativeQ5KMatmulKernel : Q5KMatmulKernel {

    private const val BLOCK_SIZE = 256
    private const val BYTES_PER_BLOCK = 176

    fun isAvailable(): Boolean = handle != null

    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "NativeQ5KMatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0 || inputDim == 0) return
        val mh = handle
            ?: error("NativeQ5KMatmulKernel.matmul invoked while native library unavailable")

        Arena.ofConfined().use { arena ->
            val inSeg = arena.allocate(
                inputDim.toLong() * java.lang.Float.BYTES,
                ValueLayout.JAVA_FLOAT.byteAlignment(),
            )
            val outSeg = arena.allocate(
                outputDim.toLong() * java.lang.Float.BYTES,
                ValueLayout.JAVA_FLOAT.byteAlignment(),
            )
            val weightBytesUsed = ((inputDim / BLOCK_SIZE).toLong() * outputDim) * BYTES_PER_BLOCK.toLong()
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
        val symbol = lookup.find("skainet_q5k_matmul").orElse(null) ?: return@lazy null
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
