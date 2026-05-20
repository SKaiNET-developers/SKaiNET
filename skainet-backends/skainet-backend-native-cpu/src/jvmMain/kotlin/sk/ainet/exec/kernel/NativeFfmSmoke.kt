package sk.ainet.exec.kernel

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.lang.invoke.MethodHandle

/**
 * End-to-end smoke test of the FFM downcall pipeline used by the
 * native kernel provider. Calls the bundled native function
 *
 *   void skainet_smoke_double(const float* input, float* output, int32_t length);
 *
 * which writes `output[i] = 2.0f * input[i]`. This object exists only
 * to validate the loader → Linker → MethodHandle path on real hardware
 * before any production kernel ships in PR 2.
 *
 * Not part of the public SPI: `internal` visibility, exposed to tests
 * via the same package.
 */
internal object NativeFfmSmoke {

    fun isAvailable(): Boolean = handle != null

    /**
     * Run the bundled smoke kernel on [input] and return a fresh
     * `FloatArray` with `output[i] = 2.0f * input[i]`. Returns `null`
     * when the native lib failed to load (callers fall back to a
     * pure-Kotlin reference).
     */
    fun double(input: FloatArray): FloatArray? {
        val mh = handle ?: return null
        val output = FloatArray(input.size)
        val byteSize = input.size.toLong() * java.lang.Float.BYTES
        val byteAlign = ValueLayout.JAVA_FLOAT.byteAlignment()
        Arena.ofConfined().use { arena ->
            val inSeg = arena.allocate(byteSize, byteAlign)
            val outSeg = arena.allocate(byteSize, byteAlign)
            MemorySegment.copy(input, 0, inSeg, ValueLayout.JAVA_FLOAT, 0L, input.size)
            mh.invoke(inSeg, outSeg, input.size)
            MemorySegment.copy(outSeg, ValueLayout.JAVA_FLOAT, 0L, output, 0, output.size)
        }
        return output
    }

    private val handle: MethodHandle? by lazy {
        val lookup = NativeLibraryLoader.lookup() ?: return@lazy null
        val symbol = lookup.find("skainet_smoke_double").orElse(null) ?: return@lazy null
        val descriptor = FunctionDescriptor.ofVoid(
            ValueLayout.ADDRESS,
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT,
        )
        runCatching { Linker.nativeLinker().downcallHandle(symbol, descriptor) }.getOrNull()
    }
}
