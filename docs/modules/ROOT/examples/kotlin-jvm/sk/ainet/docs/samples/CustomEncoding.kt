package sk.ainet.docs.samples

import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.KernelKey
import sk.ainet.backend.api.kernel.ViewKernel
import sk.ainet.lang.memory.BlockDecoder
import sk.ainet.lang.memory.Scope
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.storage.MemoryTracker
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import kotlin.math.abs

/**
 * An invented low-bit encoding, defined outside the library: one sign bit per weight and one FP32
 * scale per block of 32 weights. Every tagged region is included verbatim into
 * `how-to/custom-encoding.adoc`; `CustomEncodingTest` runs it.
 */
object CustomEncoding {

    // tag::geometry[]
    /** 32 weights per block: a 4-byte FP32 scale, then 32 sign bits. 8 bytes, so 2 bits per weight. */
    const val BLOCK: Int = 32
    const val BYTES_PER_BLOCK: Int = 8

    /**
     * `TensorEncoding` is sealed, so an encoding from outside the library is an `Opaque` one: a
     * name and the byte count. The name is what a kernel key and a memory report show.
     */
    fun encodingOf(elementCount: Int): TensorEncoding =
        TensorEncoding.Opaque("SIGN1", elementCount.toLong() / BLOCK * BYTES_PER_BLOCK)
    // end::geometry[]

    // tag::encode[]
    /** Each block keeps the signs and one scale: the mean absolute value of its 32 weights. */
    fun encode(values: FloatArray): ByteArray {
        require(values.size % BLOCK == 0) { "SIGN1 stores whole blocks of $BLOCK" }
        val out = ByteArray(values.size / BLOCK * BYTES_PER_BLOCK)
        for (block in 0 until values.size / BLOCK) {
            var sum = 0f
            var signs = 0
            for (i in 0 until BLOCK) {
                val v = values[block * BLOCK + i]
                sum += abs(v)
                if (v >= 0f) signs = signs or (1 shl i)
            }
            putInt(out, block * BYTES_PER_BLOCK, (sum / BLOCK).toRawBits())
            putInt(out, block * BYTES_PER_BLOCK + 4, signs)
        }
        return out
    }
    // end::encode[]

    private fun putInt(bytes: ByteArray, at: Int, value: Int) {
        for (i in 0 until 4) bytes[at + i] = (value ushr (8 * i)).toByte()
    }

    private fun getInt(bytes: ByteArray, at: Int): Int {
        var v = 0
        for (i in 0 until 4) v = v or ((bytes[at + i].toInt() and 0xFF) shl (8 * i))
        return v
    }

    // tag::decoder[]
    /**
     * The one thing a view needs to read the encoding: decode a block out of a `Storage`. It gets
     * the storage on every call, so the same decoder reads a heap array, a direct buffer or a
     * mapped file.
     */
    object Sign1Decoder : BlockDecoder {
        override val blockSize: Int = BLOCK
        override val bytesPerBlock: Int = BYTES_PER_BLOCK

        override fun decodeBlock(storage: Storage, blockIndex: Long, out: FloatArray, outOffset: Int) {
            val block = ByteArray(BYTES_PER_BLOCK)
            storage.copyInto(block, 0, blockIndex * BYTES_PER_BLOCK, BYTES_PER_BLOCK)
            val scale = Float.fromBits(getInt(block, 0))
            val signs = getInt(block, 4)
            for (i in 0 until BLOCK) out[outOffset + i] = if ((signs ushr i) and 1 == 1) scale else -scale
        }
    }
    // end::decoder[]

    // tag::view[]
    /** A view over the packed bytes. From here on it is a tensor view like any other. */
    fun viewOf(bytes: ByteArray, rows: Int, cols: Int): TensorView =
        TensorView.packed(
            storage = Storage.Heap.wrap(bytes, mutable = false),
            shape = Shape(rows, cols),
            encoding = encodingOf(rows * cols),
            decoder = Sign1Decoder,
        )
    // end::view[]

    // tag::kernel[]
    /**
     * Optional: a kernel that multiplies on the packed bytes. Without it the dispatcher uses its
     * decoding reference kernel, which is correct for any view that has a decoder, and slow.
     *
     * A sign-bit weight needs no multiply per element: add or subtract the activation, then scale
     * once per block.
     */
    class Sign1MatmulKernel(override val key: KernelKey) : ViewKernel {
        override val name: String = "sample-SIGN1"

        override fun run(inputs: List<TensorView>, out: TensorView) {
            val (activation, weight) = inputs
            val x = activation.toFloatArray()
            val (rows, cols) = weight.shape.dimensions
            val block = ByteArray(BYTES_PER_BLOCK)
            for (r in 0 until rows) {
                var sum = 0f
                for (b in 0 until cols / BLOCK) {
                    weight.storage.copyInto(block, 0, (r.toLong() * (cols / BLOCK) + b) * BYTES_PER_BLOCK, BYTES_PER_BLOCK)
                    val signs = getInt(block, 4)
                    var acc = 0f
                    for (i in 0 until BLOCK) {
                        val v = x[b * BLOCK + i]
                        acc += if ((signs ushr i) and 1 == 1) v else -v
                    }
                    sum += acc * Float.fromBits(getInt(block, 0))
                }
                out.set(0, r, value = sum)
            }
        }
    }

    /** Register the kernel under the key of the operand pair it serves. */
    fun installKernel(activation: TensorView, weight: TensorView) {
        // Let the library install its own kernels first: a table that is not empty is taken as
        // "someone bootstrapped", and the built-in packs would then never be installed.
        KernelDispatch.ensureInstalled()
        val key = KernelKey.matmul(activation, weight)
        if (KernelDispatch.find(key) == null) KernelDispatch.register(Sign1MatmulKernel(key))
    }
    // end::kernel[]

    class Result(
        val format: String,
        val bitsPerWeight: Double,
        val compressionRatio: Double,
        val decoded: FloatArray,
        val firstElement: Float,
        val materializedFormat: String,
        val referenceKernel: String,
        val referenceProduct: FloatArray,
        val kernel: String,
        val product: FloatArray,
        val keyText: String,
    )

    fun run(rows: Int = 4, cols: Int = 64): Result {
        val values = FloatArray(rows * cols) { i -> ((i * 37 % 19) - 9) / 9f }

        // tag::use[]
        val weight = viewOf(encode(values), rows, cols)

        val format = weight.format.toString()              // Float32/SIGN1
        val first = weight.get(0, 0)                       // decodes one element
        val decoded = weight.toFloatArray()                // decodes all of them
        val dense = weight.materialize()                   // a dense FP32 copy, if one is needed

        val tracker = MemoryTracker()
        tracker.record("w", weight)
        val entry = tracker.report().entries.single().report
        val bitsPerWeight = entry.physicalBytes * 8.0 / (rows * cols)
        // end::use[]

        // tag::matmul[]
        val x = FloatArray(cols) { i -> ((i % 5) - 2) / 2f }
        val activation = TensorView.dense(Storage.Heap.wrap(x), Shape(1, cols), FP32)

        // Before the kernel is registered: the decoding reference kernel serves the pair.
        val before = RecordingTraceSink()
        val outBefore = TensorView.dense(Storage.Heap.floats(rows), Shape(1, rows), FP32)
        KernelDispatch.matmul(activation, weight, outBefore, Scope.Ambient, before)

        // After: the same call selects the kernel by the key of the two views.
        installKernel(activation, weight)
        val after = RecordingTraceSink()
        val outAfter = TensorView.dense(Storage.Heap.floats(rows), Shape(1, rows), FP32)
        KernelDispatch.matmul(activation, weight, outAfter, Scope.Ambient, after)
        // end::matmul[]

        return Result(
            format = format,
            bitsPerWeight = bitsPerWeight,
            compressionRatio = entry.compressionRatio,
            decoded = decoded,
            firstElement = first,
            materializedFormat = dense.format.toString(),
            referenceKernel = before.eventsOf<TraceEvent.KernelRun>().single().kernel,
            referenceProduct = FloatArray(rows) { outBefore.get(0, it) },
            kernel = after.eventsOf<TraceEvent.KernelRun>().single().kernel,
            product = FloatArray(rows) { outAfter.get(0, it) },
            keyText = KernelKey.matmul(activation, weight).toString(),
        )
    }
}
