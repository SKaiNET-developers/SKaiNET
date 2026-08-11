package sk.ainet.io.gguf

import kotlinx.coroutines.runBlocking
import kotlinx.io.buffered
import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import java.io.File
import java.lang.management.ManagementFactory
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Heap-instrumented regression gate for #782 (GGUF DEQUANTIZE_TO_FP32
 * over-allocation).
 *
 * Measures *bytes allocated on the loading thread* via the JVM's per-thread
 * allocation counter — deterministic, unlike sampling peak heap around GC.
 * The legit cost of a full FP32 materialization is:
 *
 *   dense FP32 destination + packed source bytes (streamed per tensor)
 *
 * The historical path additionally built a full-size dequant intermediate and
 * a full-size defensive factory copy per tensor (~2x the FP32 total in
 * transients, ~3x peak with the boxed legacy reader — the ">12 GB for a
 * 4.4 GB model" in the issue). The assertions here pin the fixed path to a
 * *transient* budget of a fraction of the FP32 total.
 */
class DequantHeapUsageTest {

    private val threadMx = ManagementFactory.getThreadMXBean() as com.sun.management.ThreadMXBean

    private fun allocatedBytes(): Long = threadMx.getThreadAllocatedBytes(Thread.currentThread().id)

    private fun buildModel(): Array<SyntheticGguf.TestTensor> = arrayOf(
        // dominated by K-quants, like a real Q4_K_M file
        SyntheticGguf.tensor("blk0.q4k", GGMLQuantizationType.Q4_K, elements = 4 * 1024 * 1024),
        SyntheticGguf.tensor("blk1.q4k", GGMLQuantizationType.Q4_K, elements = 2 * 1024 * 1024),
        SyntheticGguf.tensor("blk2.q6k", GGMLQuantizationType.Q6_K, elements = 2 * 1024 * 1024),
        SyntheticGguf.tensor("blk3.q80", GGMLQuantizationType.Q8_0, elements = 1024 * 1024),
        SyntheticGguf.tensor("blk4.f16", GGMLQuantizationType.F16, elements = 1024 * 1024),
        SyntheticGguf.tensor("blk5.f32", GGMLQuantizationType.F32, elements = 512 * 1024),
    )

    @Test
    fun `streaming DEQUANTIZE_TO_FP32 stays within the transient allocation budget`() {
        val model = buildModel()
        val fp32Total = model.sumOf { it.elementCount * 4L }
        val rawTotal = model.sumOf { it.data.size.toLong() }
        val file = SyntheticGguf.write(*model)
        try {
            // Warm-up on a tiny file: classloading, JIT, coroutine machinery.
            val warmup = SyntheticGguf.write(
                SyntheticGguf.tensor("w.q4k", GGMLQuantizationType.Q4_K, elements = 512),
            )
            loadAll(warmup, QuantPolicy.DEQUANTIZE_TO_FP32)
            warmup.delete()

            val before = allocatedBytes()
            val loaded = loadAll(file, QuantPolicy.DEQUANTIZE_TO_FP32)
            val allocated = allocatedBytes() - before

            // Keep the result alive so the resident set is real.
            assertTrue(loaded.size == model.size)

            val transient = allocated - fp32Total
            println(
                "DEQUANTIZE_TO_FP32 streaming load: fp32Total=${mb(fp32Total)} MB, " +
                    "rawTotal=${mb(rawTotal)} MB, allocated=${mb(allocated)} MB, " +
                    "transient=${mb(transient)} MB (${"%.2f".format(allocated / fp32Total.toDouble())}x of FP32 total)",
            )

            // Budget: packed source bytes (streamed, one tensor at a time) plus
            // per-block scratch and metadata — but *no* full-size FP32 copies.
            // 1.2x of the FP32 total is the issue's acceptance bar; the fixed
            // path lands well under it.
            val budget = (0.2 * fp32Total).toLong() + rawTotal + 16L * 1024 * 1024
            assertTrue(
                transient <= budget,
                "transient allocation ${mb(transient)} MB exceeds budget ${mb(budget)} MB " +
                    "(allocated ${mb(allocated)} MB vs FP32 total ${mb(fp32Total)} MB) — " +
                    "a full-size intermediate or defensive copy is back on the load path (#782)",
            )
        } finally {
            file.delete()
        }
    }

    /**
     * Comparative measurement of the historical copy chain (dequant intermediate
     * + factory defensive copy) — printed for the record, and asserted to be
     * strictly worse than the fixed path so this test documents *why* the loader
     * wraps instead of copies.
     */
    @Test
    fun `historical copy chain allocates roughly twice the FP32 total`() {
        val model = buildModel().filter { it.type != GGMLQuantizationType.F32 && it.type != GGMLQuantizationType.F16 }
        val fp32Total = model.sumOf { it.elementCount * 4L }
        val ctx = DefaultDataExecutionContext()

        // warm-up
        run {
            val t = SyntheticGguf.tensor("w.q4k", GGMLQuantizationType.Q4_K, elements = 512)
            val floats = DequantOps.dequantFromBytes(t.data, t.type, t.elementCount.toInt())
            ctx.fromFloatArray<FP32, Float>(Shape(floats.size), FP32::class, floats)
        }

        val kept = ArrayList<Tensor<FP32, Float>>(model.size)
        val before = allocatedBytes()
        for (t in model) {
            // pre-fix sequence: full-size intermediate, then factory copy
            val floats = DequantOps.dequantFromBytes(t.data, t.type, t.elementCount.toInt())
            kept += ctx.fromFloatArray(Shape(floats.size), FP32::class, floats)
        }
        val allocated = allocatedBytes() - before
        assertTrue(kept.size == model.size)

        println(
            "historical copy chain: fp32Total=${mb(fp32Total)} MB, allocated=${mb(allocated)} MB " +
                "(${"%.2f".format(allocated / fp32Total.toDouble())}x of FP32 total)",
        )
        assertTrue(
            allocated >= 2 * fp32Total,
            "expected the historical chain to allocate >= 2x the FP32 total " +
                "(intermediate + defensive copy); measured ${mb(allocated)} MB",
        )
    }

    /**
     * The legacy in-memory [GGUFReader] used to box every payload element of
     * every tensor at parse time (one object per byte for quantized payloads —
     * the >12 GB transient for a 637 MB file in #782). After the lazy-view fix
     * its parse-time allocation is O(file size), not O(boxed elements).
     */
    @Test
    fun `legacy GGUFReader parse allocates O of file size, not O of boxed elements`() {
        val model = buildModel()
        val file = SyntheticGguf.write(*model)
        val fileBytes = file.length()
        try {
            // warm-up
            SyntheticGguf.write(SyntheticGguf.tensor("w.q80", GGMLQuantizationType.Q8_0, elements = 1024)).let {
                GGUFReader(sourceOf(it), loadTensorData = true)
                it.delete()
            }

            val before = allocatedBytes()
            val reader = GGUFReader(sourceOf(file), loadTensorData = true)
            val allocated = allocatedBytes() - before
            assertTrue(reader.tensors.size == model.size)

            println(
                "legacy GGUFReader parse: file=${mb(fileBytes)} MB, allocated=${mb(allocated)} MB " +
                    "(${"%.2f".format(allocated / fileBytes.toDouble())}x of file size)",
            )
            // Pre-fix this was ~20-30x the payload size (boxed element objects
            // plus chunked() garbage); post-fix it is the file buffer plus
            // metadata.
            assertTrue(
                allocated <= 3 * fileBytes + 16L * 1024 * 1024,
                "legacy GGUFReader parse allocated ${mb(allocated)} MB for a ${mb(fileBytes)} MB file — " +
                    "eager boxed materialization is back (#782)",
            )

            // Lazy views must still deliver correct payloads: spot-check the F32
            // tensor against its little-endian encoding.
            val f32 = reader.tensors.first { it.tensorType == GGMLQuantizationType.F32 }
            val payload = model.first { it.type == GGMLQuantizationType.F32 }.data
            val view = reader.materialize(f32)
            assertTrue(view.size == f32.nElements)
            for (i in intArrayOf(0, 1, view.size / 2, view.size - 1)) {
                val bits = (payload[i * 4].toInt() and 0xFF) or
                    ((payload[i * 4 + 1].toInt() and 0xFF) shl 8) or
                    ((payload[i * 4 + 2].toInt() and 0xFF) shl 16) or
                    ((payload[i * 4 + 3].toInt() and 0xFF) shl 24)
                assertTrue(
                    (view[i] as Float).toRawBits() == bits,
                    "lazy F32 view mismatch at $i",
                )
            }
        } finally {
            file.delete()
        }
    }

    private fun sourceOf(file: File): kotlinx.io.Source =
        kotlinx.io.files.SystemFileSystem.source(kotlinx.io.files.Path(file.absolutePath)).buffered()

    private fun loadAll(file: File, policy: QuantPolicy): Map<String, Tensor<FP32, Float>> {
        val ctx = DefaultDataExecutionContext()
        val loaded = mutableMapOf<String, Tensor<FP32, Float>>()
        runBlocking {
            StreamingGgufParametersLoader(
                sourceProvider = { JvmRandomAccessSource.open(file) },
                quantPolicy = policy,
            ).load<FP32, Float>(ctx, FP32::class) { name, tensor -> loaded[name] = tensor }
        }
        return loaded
    }

    private fun mb(bytes: Long): Long = bytes / (1024 * 1024)
}
