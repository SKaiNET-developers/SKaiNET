package sk.ainet.exec.kernel

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle
import sk.ainet.backend.api.kernel.Q4_0MatmulKernel

/**
 * Native (FFM) implementation of [Q4_0MatmulKernel].
 *
 * Wraps the bundled C symbol
 *
 *   void skainet_q4_0_matmul(
 *       const float* input,    int32_t input_offset,
 *       const uint8_t* weight, int32_t weight_byte_offset,
 *       int32_t input_dim,     int32_t output_dim,
 *       float* output,         int32_t output_offset);
 *
 * The C kernel decodes the ggml-canonical Q4_0 block (FP16 scale + 16
 * packed bytes, split nibble layout) with `(code - 8) * d` dequant and a
 * tight inner FMA the compiler auto-vectorizes under -O3 -ffast-math.
 *
 * Numerical parity vs [ScalarQ4_0MatmulKernel] is asserted by
 * `NativeQ4_0MatmulKernelParityTest` within the same `1e-2 *
 * blocksPerInputDim` band the Panama parity uses.
 */
internal object NativeQ4_0MatmulKernel : Q4_0MatmulKernel {

    fun isAvailable(): Boolean = handle != null

    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: ByteArray, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "NativeQ4_0MatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        if (outputDim == 0) return

        val mh = handle
            ?: error("NativeQ4_0MatmulKernel.matmul invoked while native library unavailable")

        val blocksPerInputDim = inputDim / BLOCK_SIZE
        val inputReachFloats = if (inputDim == 0) 0 else inputOffset + inputDim
        val weightReachBytes = if (inputDim == 0 || outputDim == 0) 0
                               else weightByteOffset + blocksPerInputDim * outputDim * BYTES_PER_BLOCK
        val outputReachFloats = outputOffset + outputDim

        Arena.ofConfined().use { arena ->
            val fAlign = ValueLayout.JAVA_FLOAT.byteAlignment()
            val bAlign = ValueLayout.JAVA_BYTE.byteAlignment()

            val inputSeg: MemorySegment = if (inputReachFloats > 0)
                arena.allocate(inputReachFloats.toLong() * java.lang.Float.BYTES, fAlign)
            else MemorySegment.NULL
            val weightSeg: MemorySegment = if (weightReachBytes > 0)
                arena.allocate(weightReachBytes.toLong(), bAlign)
            else MemorySegment.NULL
            val outputSeg: MemorySegment =
                arena.allocate(outputReachFloats.toLong() * java.lang.Float.BYTES, fAlign)

            if (inputReachFloats > 0) {
                MemorySegment.copy(input, 0, inputSeg, ValueLayout.JAVA_FLOAT, 0L, inputReachFloats)
            }
            if (weightReachBytes > 0) {
                MemorySegment.copy(weight, 0, weightSeg, ValueLayout.JAVA_BYTE, 0L, weightReachBytes)
            }

            mh.invoke(
                inputSeg, inputOffset,
                weightSeg, weightByteOffset,
                inputDim, outputDim,
                outputSeg, outputOffset,
            )

            MemorySegment.copy(outputSeg, ValueLayout.JAVA_FLOAT, 0L, output, 0, outputReachFloats)
        }
    }

    private const val BLOCK_SIZE = 32
    private const val BYTES_PER_BLOCK = 18

    private val handle: MethodHandle? by lazy {
        val lookup = NativeLibraryLoader.lookup() ?: return@lazy null
        val symbol = lookup.find("skainet_q4_0_matmul").orElse(null) ?: return@lazy null
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
