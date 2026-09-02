package sk.ainet.io.safetensors

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Bf16DenseTensorData
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.Fp16DenseTensorData
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP32

/**
 * End-to-end coverage of [ShardedSafeTensorsParametersLoader] against a
 * genuine 2-shard fixture (`model-0000X-of-00002.safetensors` + index),
 * plus the 1-shard-index vs single-file parity test that guards the
 * [SafeTensorsMaterializer] extraction refactor.
 *
 * Fixture layout:
 * - shard 1: `alpha.weight` (F32 2×2), `bravo.weight` (BF16 4)
 * - shard 2: `charlie.weight` (F16 4), `zulu.weight` (F32 3)
 * Name-sorted delivery order is therefore alpha, bravo, charlie, zulu —
 * interleaving would differ if delivery were shard-ordered only.
 */
class ShardedSafeTensorsParametersLoaderJvmTest {

    private val bf16AbsTol = 1e-2f

    private val alphaValues = floatArrayOf(1f, 2f, 3f, 4f)
    private val bravoValues = floatArrayOf(0.5f, -1.0f, 2.5f, -64.0f)
    private val charlieValues = floatArrayOf(1.0f, -2.0f, 0.25f, 8.0f) // exactly representable in F16
    private val zuluValues = floatArrayOf(-7f, 0f, 42f)

    // ---- fixture builders ----

    private fun fp32ToBf16Bytes(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bf16 = (values[i].toRawBits() ushr 16) and 0xFFFF
            out[i * 2] = (bf16 and 0xFF).toByte()
            out[i * 2 + 1] = ((bf16 ushr 8) and 0xFF).toByte()
        }
        return out
    }

    /** Normal-range-only FP32 → F16 encoder; test values must be exactly representable. */
    private fun fp32ToF16Bytes(values: FloatArray): ByteArray {
        val out = ByteArray(values.size * 2)
        for (i in values.indices) {
            val bits = values[i].toRawBits()
            val sign = (bits ushr 16) and 0x8000
            val half = if (bits and 0x7FFFFFFF == 0) {
                sign // ±0
            } else {
                val exp32 = (bits ushr 23) and 0xFF
                val mant = bits and 0x7FFFFF
                val expH = exp32 - 127 + 15
                check(expH in 1..30) { "test value ${values[i]} not a normal F16" }
                check(mant and 0x1FFF == 0) { "test value ${values[i]} not exact in F16" }
                sign or (expH shl 10) or (mant ushr 13)
            }
            out[i * 2] = (half and 0xFF).toByte()
            out[i * 2 + 1] = ((half ushr 8) and 0xFF).toByte()
        }
        return out
    }

    private fun fp32Bytes(values: FloatArray): ByteArray {
        val buf = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach { buf.putFloat(it) }
        return buf.array()
    }

    private fun writeShard(path: Path, entries: List<Triple<String, String, Pair<List<Long>, ByteArray>>>) {
        var offset = 0L
        val headerJson = entries.joinToString(prefix = "{", postfix = "}", separator = ",") { (name, dtype, shapeAndBytes) ->
            val (shape, bytes) = shapeAndBytes
            val start = offset
            offset += bytes.size
            "\"$name\": {\"dtype\": \"$dtype\", \"shape\": [${shape.joinToString(", ")}], " +
                "\"data_offsets\": [$start, $offset]}"
        }
        val headerBytes = headerJson.toByteArray(Charsets.UTF_8)
        path.toFile().outputStream().use { out ->
            out.write(
                ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
                    .putLong(headerBytes.size.toLong()).array(),
            )
            out.write(headerBytes)
            entries.forEach { (_, _, shapeAndBytes) -> out.write(shapeAndBytes.second) }
        }
    }

    /**
     * Write the standard 2-shard fixture into [dir]; returns the index path.
     * Set [omitShard2] to leave shard 2 missing on disk (still referenced by the index).
     */
    private fun writeTwoShardFixture(dir: Path, omitShard2: Boolean = false): String {
        val shard1 = "model-00001-of-00002.safetensors"
        val shard2 = "model-00002-of-00002.safetensors"
        writeShard(
            dir.resolve(shard1),
            listOf(
                Triple("alpha.weight", "F32", listOf(2L, 2L) to fp32Bytes(alphaValues)),
                Triple("bravo.weight", "BF16", listOf(4L) to fp32ToBf16Bytes(bravoValues)),
            ),
        )
        if (!omitShard2) {
            writeShard(
                dir.resolve(shard2),
                listOf(
                    Triple("charlie.weight", "F16", listOf(4L) to fp32ToF16Bytes(charlieValues)),
                    Triple("zulu.weight", "F32", listOf(3L) to fp32Bytes(zuluValues)),
                ),
            )
        }
        val totalSize = Files.size(dir.resolve(shard1)) +
            (if (omitShard2) 0L else Files.size(dir.resolve(shard2)))
        val indexPath = dir.resolve("model.safetensors.index.json")
        Files.writeString(
            indexPath,
            """
            {
              "metadata": {"total_size": $totalSize},
              "weight_map": {
                "alpha.weight": "$shard1",
                "bravo.weight": "$shard1",
                "charlie.weight": "$shard2",
                "zulu.weight": "$shard2"
              }
            }
            """.trimIndent(),
        )
        return indexPath.toString()
    }

    private fun loadAll(loader: ShardedSafeTensorsParametersLoader): Pair<List<String>, Map<String, Tensor<FP32, Float>>> =
        runBlocking {
            val ctx = DirectCpuExecutionContext.create()
            val order = mutableListOf<String>()
            val out = mutableMapOf<String, Tensor<FP32, Float>>()
            loader.load<FP32, Float>(ctx, FP32::class) { name, tensor ->
                order.add(name)
                out[name] = tensor
            }
            order to out
        }

    private fun assertClose(expected: FloatArray, actual: FloatArray, tol: Float, label: String) {
        assertEquals(expected.size, actual.size, "$label size")
        for (i in expected.indices) {
            assertTrue(
                abs(expected[i] - actual[i]) <= tol,
                "$label mismatch at $i: expected=${expected[i]} actual=${actual[i]}",
            )
        }
    }

    // ---- tests ----

    @Test
    fun `delivers all tensors across both shards name-sorted with exact values`() {
        val dir = Files.createTempDirectory("sharded-loader-")
        try {
            val indexPath = writeTwoShardFixture(dir)
            val (order, tensors) = loadAll(ShardedSafeTensorsParametersLoader(indexPath))

            assertEquals(listOf("alpha.weight", "bravo.weight", "charlie.weight", "zulu.weight"), order)
            assertContentEquals(alphaValues, tensors["alpha.weight"]!!.data.copyToFloatArray(), "alpha exact")
            assertContentEquals(zuluValues, tensors["zulu.weight"]!!.data.copyToFloatArray(), "zulu exact")
            assertClose(bravoValues, tensors["bravo.weight"]!!.data.copyToFloatArray(), bf16AbsTol, "bravo")
            // F16 fixture values are exactly representable — zero tolerance.
            assertContentEquals(charlieValues, tensors["charlie.weight"]!!.data.copyToFloatArray(), "charlie exact")
            assertEquals(listOf(2, 2), tensors["alpha.weight"]!!.shape.dimensions.toList())
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun `default policy dequants narrow floats and keep_native preserves storage types`() {
        val dir = Files.createTempDirectory("sharded-loader-policy-")
        try {
            val indexPath = writeTwoShardFixture(dir)

            val (_, dequanted) = loadAll(ShardedSafeTensorsParametersLoader(indexPath))
            assertTrue(dequanted["bravo.weight"]!!.data is FloatArrayTensorData<*>, "default BF16 → FloatArray")
            assertTrue(dequanted["charlie.weight"]!!.data is FloatArrayTensorData<*>, "default F16 → FloatArray")

            val (_, kept) = loadAll(
                ShardedSafeTensorsParametersLoader(
                    indexPath = indexPath,
                    bf16Policy = Bf16LoadPolicy.KEEP_NATIVE,
                    fp16Policy = NarrowFloatLoadPolicy.KEEP_NATIVE,
                ),
            )
            assertTrue(kept["bravo.weight"]!!.data is Bf16DenseTensorData, "KEEP_NATIVE BF16 storage")
            assertTrue(kept["charlie.weight"]!!.data is Fp16DenseTensorData, "KEEP_NATIVE F16 storage")
            assertTrue(kept["alpha.weight"]!!.data is FloatArrayTensorData<*>, "F32 unaffected by policies")
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun `withPolicy routes Require BF16 to keep_native for bf16 only`() {
        val dir = Files.createTempDirectory("sharded-loader-withpolicy-")
        try {
            val indexPath = writeTwoShardFixture(dir)
            val (_, tensors) = loadAll(
                ShardedSafeTensorsParametersLoader.withPolicy(indexPath, DTypePolicy.Require(BF16)),
            )
            assertTrue(tensors["bravo.weight"]!!.data is Bf16DenseTensorData, "Require(BF16) keeps BF16 native")
            assertTrue(
                tensors["charlie.weight"]!!.data is FloatArrayTensorData<*>,
                "Require(BF16) must not keep F16 native",
            )
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun `tensorFilter excludes tensors from delivery and progress total`() {
        val dir = Files.createTempDirectory("sharded-loader-filter-")
        try {
            val indexPath = writeTwoShardFixture(dir)
            val totals = mutableSetOf<Long>()
            val loader = ShardedSafeTensorsParametersLoader(
                indexPath = indexPath,
                onProgress = { _, total, _ -> totals.add(total) },
                tensorFilter = { it.name != "bravo.weight" },
            )
            val (order, tensors) = loadAll(loader)
            assertEquals(listOf("alpha.weight", "charlie.weight", "zulu.weight"), order)
            assertTrue("bravo.weight" !in tensors)
            assertEquals(setOf(3L), totals, "progress total must reflect the filtered count")
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun `missing shard throws IncompleteShard unless allowPartial delivers shard-1 tensors only`() {
        val dir = Files.createTempDirectory("sharded-loader-partial-")
        try {
            val indexPath = writeTwoShardFixture(dir, omitShard2 = true)

            assertFailsWith<SafeTensorsShardException.IncompleteShard> {
                loadAll(ShardedSafeTensorsParametersLoader(indexPath))
            }

            val (order, tensors) = loadAll(
                ShardedSafeTensorsParametersLoader(indexPath, allowPartial = true),
            )
            assertEquals(listOf("alpha.weight", "bravo.weight"), order)
            assertContentEquals(alphaValues, tensors["alpha.weight"]!!.data.copyToFloatArray())
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun `fail-fast dtype mismatch throws before any tensor is delivered`() {
        val dir = Files.createTempDirectory("sharded-loader-failfast-")
        try {
            val shard = "model-00001-of-00001.safetensors"
            val intBytes = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
                .putInt(7).putInt(9).array()
            writeShard(
                dir.resolve(shard),
                listOf(
                    Triple("good.weight", "F32", listOf(2L) to fp32Bytes(floatArrayOf(1f, 2f))),
                    Triple("ids.tokens", "I32", listOf(2L) to intBytes),
                ),
            )
            val indexPath = dir.resolve("model.safetensors.index.json")
            Files.writeString(
                indexPath,
                """{"metadata":{"total_size":${Files.size(dir.resolve(shard))}},""" +
                    """"weight_map":{"good.weight":"$shard","ids.tokens":"$shard"}}""",
            )

            var delivered = 0
            val error = assertFailsWith<IllegalArgumentException> {
                runBlocking {
                    val ctx = DirectCpuExecutionContext.create()
                    ShardedSafeTensorsParametersLoader(indexPath.toString())
                        .load<FP32, Float>(ctx, FP32::class) { _, _ -> delivered++ }
                }
            }
            assertEquals(0, delivered, "fail-fast must fire before the first onTensorLoaded")
            assertTrue("ids.tokens" in (error.message ?: ""), "error must name the offending tensor")
        } finally {
            dir.toFile().deleteRecursively()
        }
    }

    @Test
    fun `one-shard index parity with single-file loader`() {
        // Guards the SafeTensorsMaterializer extraction: identical bytes through
        // either loader must yield identical tensors (values and storage types).
        val dir = Files.createTempDirectory("sharded-loader-parity-")
        try {
            val shard = "model.safetensors"
            writeShard(
                dir.resolve(shard),
                listOf(
                    Triple("alpha.weight", "F32", listOf(2L, 2L) to fp32Bytes(alphaValues)),
                    Triple("bravo.weight", "BF16", listOf(4L) to fp32ToBf16Bytes(bravoValues)),
                    Triple("charlie.weight", "F16", listOf(4L) to fp32ToF16Bytes(charlieValues)),
                ),
            )
            val indexPath = dir.resolve("model.safetensors.index.json")
            Files.writeString(
                indexPath,
                """{"metadata":{"total_size":${Files.size(dir.resolve(shard))}},""" +
                    """"weight_map":{"alpha.weight":"$shard","bravo.weight":"$shard","charlie.weight":"$shard"}}""",
            )

            val singleFile: Map<String, Tensor<FP32, Float>> = runBlocking {
                val ctx = DirectCpuExecutionContext.create()
                val out = mutableMapOf<String, Tensor<FP32, Float>>()
                SafeTensorsParametersLoader(
                    sourceProvider = { JvmRandomAccessSource.open(File(dir.toFile(), shard)) },
                ).load<FP32, Float>(ctx, FP32::class) { name, tensor -> out[name] = tensor }
                out
            }
            val (_, sharded) = loadAll(ShardedSafeTensorsParametersLoader(indexPath.toString()))

            assertEquals(singleFile.keys, sharded.keys)
            for ((name, single) in singleFile) {
                val fromSharded = sharded[name]!!
                assertEquals(
                    single.data::class, fromSharded.data::class,
                    "storage type parity for '$name'",
                )
                assertContentEquals(
                    single.data.copyToFloatArray(), fromSharded.data.copyToFloatArray(),
                    "value parity for '$name'",
                )
            }
        } finally {
            dir.toFile().deleteRecursively()
        }
    }
}
