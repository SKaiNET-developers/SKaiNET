package sk.ainet.exec.kernel

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle
import sk.ainet.backend.api.kernel.Q4KMemSegMatmulKernel

/**
 * Zero-copy native [Q4KMemSegMatmulKernel] implementation.
 *
 * Reuses the same `skainet_q4k_matmul` C symbol as
 * [NativeQ4KMatmulKernel] — the C side just sees `const uint8_t*` and
 * doesn't care whether the Kotlin caller backed those bytes by a
 * staged copy of a `ByteArray` or by an mmap'd off-heap segment. The
 * win on this path is that the weight bytes (which dominate the
 * payload — typical LLM Q4_K tensor: tens to hundreds of MB per layer)
 * never round-trip through the heap.
 *
 * Per-call cost vs [NativeQ4KMatmulKernel]:
 *  - skips `MemorySegment.copy(weight, ...)` of `inputDim/256 * outputDim
 *    * 144` bytes (e.g. 9 MB at 4096² shape).
 *  - still copies `inputDim * 4` bytes for the input vector and
 *    `outputDim * 4` bytes for the output — the input/output are
 *    typically heap arrays produced/consumed by the surrounding
 *    forward pass.
 *
 * PR 3 of the staged native-FFM rollout — see the `native-ffm-plan`
 * asciidoc.
 */
internal object NativeQ4KMemSegMatmulKernel : Q4KMemSegMatmulKernel {

    private const val BLOCK_SIZE = 256

    fun isAvailable(): Boolean = handle != null

    override fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: MemorySegment, weightByteOffset: Long,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) {
        require(inputDim % BLOCK_SIZE == 0) {
            "NativeQ4KMemSegMatmulKernel: inputDim must be a multiple of $BLOCK_SIZE; got $inputDim"
        }
        require(weightByteOffset >= 0) {
            "NativeQ4KMemSegMatmulKernel: weightByteOffset must be non-negative; got $weightByteOffset"
        }
        require(weightByteOffset <= Int.MAX_VALUE) {
            "NativeQ4KMemSegMatmulKernel: weightByteOffset $weightByteOffset exceeds Int range — " +
                "the C kernel takes int32_t today; slice the segment first or wait for the int64_t overload"
        }
        if (outputDim == 0 || inputDim == 0) return
        val mh = handle
            ?: error("NativeQ4KMemSegMatmulKernel.matmul invoked while native library unavailable")

        // The C kernel reads weight from offset 0..weightBytesUsed, so
        // require that the caller's segment is large enough. This catches
        // scope/aliasing bugs early; without it, an undersized segment
        // would crash the JVM with SIGSEGV from native code.
        val weightBytesUsed = ((inputDim / BLOCK_SIZE).toLong() * outputDim) * 144L
        require(weightByteOffset + weightBytesUsed <= weight.byteSize()) {
            "NativeQ4KMemSegMatmulKernel: weight segment too small — needs " +
                "$weightBytesUsed bytes from offset $weightByteOffset, " +
                "segment is ${weight.byteSize()} bytes"
        }

        Arena.ofConfined().use { arena ->
            val inSeg = arena.allocate(
                inputDim.toLong() * java.lang.Float.BYTES,
                ValueLayout.JAVA_FLOAT.byteAlignment(),
            )
            val outSeg = arena.allocate(
                outputDim.toLong() * java.lang.Float.BYTES,
                ValueLayout.JAVA_FLOAT.byteAlignment(),
            )
            MemorySegment.copy(input, inputOffset, inSeg, ValueLayout.JAVA_FLOAT, 0L, inputDim)

            mh.invoke(
                inSeg, 0,
                weight, weightByteOffset.toInt(),
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
            ValueLayout.ADDRESS,    // weight (passed straight through from caller)
            ValueLayout.JAVA_INT,   // weight_byte_offset
            ValueLayout.JAVA_INT,   // input_dim
            ValueLayout.JAVA_INT,   // output_dim
            ValueLayout.ADDRESS,    // output
            ValueLayout.JAVA_INT,   // output_offset
        )
        runCatching { Linker.nativeLinker().downcallHandle(symbol, descriptor) }.getOrNull()
    }
}
