package sk.ainet.io.safetensors

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import sk.ainet.io.model.DataType
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int8
import sk.ainet.lang.types.Ternary

/**
 * Unit tests for [ShardedSafeTensorsParametersLoader]'s policy routing and
 * fail-fast pre-scan. Mirrors the strategy documented in
 * [SafeTensorsParametersLoaderPolicyTest]: `withPolicy` is a thin wrapper
 * over [SafeTensorsMaterializer]'s mappers plus the constructor, so testing
 * the mappers and the pre-scan covers the routing logic without a real
 * multi-shard fixture (that lives in the jvmTest suite).
 */
class ShardedSafeTensorsParametersLoaderPolicyTest {

    // ---- Policy mapping parity with the single-file loader ----

    @Test
    fun sharded_and_single_file_loaders_share_one_policy_mapper() {
        // Both loaders must route DTypePolicy through the same mapper: spot-check
        // the four policy shapes for BF16 and FP16 arms.
        for (policy in listOf(
            DTypePolicy.Any,
            DTypePolicy.Require(BF16),
            DTypePolicy.Require(FP32),
            DTypePolicy.Prefer(BF16),
            DTypePolicy.OneOf(setOf(BF16, FP32)),
        )) {
            assertEquals(
                SafeTensorsParametersLoader.mapPolicyToBf16(policy),
                SafeTensorsMaterializer.mapPolicyToBf16(policy),
                "BF16 mapping diverged for $policy",
            )
            assertEquals(
                SafeTensorsParametersLoader.mapPolicyToFp16(policy),
                SafeTensorsMaterializer.mapPolicyToFp16(policy),
                "FP16 mapping diverged for $policy",
            )
        }
    }

    @Test
    fun require_unsatisfiable_dtype_throws() {
        assertFailsWith<IllegalArgumentException> {
            SafeTensorsMaterializer.mapPolicyToBf16(DTypePolicy.Require(Ternary))
        }
    }

    // ---- Fail-fast pre-scan ----

    private fun info(name: String, dtype: String, dataType: DataType) = ShardedTensorInfo(
        base = StreamingSafeTensorInfo(
            name = name,
            dtype = dtype,
            dataType = dataType,
            shape = listOf(2L, 2L),
            elementCount = 4L,
            dataOffsetStart = 0L,
            dataOffsetEnd = 16L,
            sizeInBytes = 16L,
            absoluteDataOffset = 8L,
        ),
        shardFilename = "model-00001-of-00002.safetensors",
        shardIndex = 1,
        totalShards = 2,
    )

    @Test
    fun pre_scan_passes_for_matching_dtypes() {
        ShardedSafeTensorsParametersLoader.failFastOnUnsupportedTensorTypes(
            listOf(
                info("a.weight", "F32", DataType.FLOAT32),
                info("b.weight", "BF16", DataType.BFLOAT16),
                info("c.weight", "F16", DataType.FLOAT16),
            ),
            FP32::class,
        )
    }

    @Test
    fun pre_scan_aggregates_all_mismatches_in_one_error() {
        val error = assertFailsWith<IllegalArgumentException> {
            ShardedSafeTensorsParametersLoader.failFastOnUnsupportedTensorTypes(
                listOf(
                    info("ok.weight", "F32", DataType.FLOAT32),
                    info("ids.tokens", "I64", DataType.INT64),
                    info("mask.bits", "BOOL", DataType.BOOL),
                ),
                FP32::class,
            )
        }
        val message = error.message ?: ""
        assertTrue("ids.tokens" in message, "expected first mismatch listed: $message")
        assertTrue("mask.bits" in message, "expected second mismatch listed: $message")
        assertTrue("2 of 3" in message, "expected aggregate count: $message")
        assertTrue("ok.weight" !in message, "matching tensor must not be listed: $message")
    }

    @Test
    fun pre_scan_names_the_required_dtype_per_tensor() {
        val error = assertFailsWith<IllegalArgumentException> {
            ShardedSafeTensorsParametersLoader.failFastOnUnsupportedTensorTypes(
                listOf(info("ids.tokens", "I32", DataType.INT32)),
                Int8::class,
            )
        }
        assertTrue("Int32" in (error.message ?: ""))
    }

    @Test
    fun required_dtype_mapping_matches_materializer_arms() {
        assertEquals(FP32::class, SafeTensorsMaterializer.requiredDType(DataType.FLOAT32))
        assertEquals(FP32::class, SafeTensorsMaterializer.requiredDType(DataType.FLOAT64))
        assertEquals(FP32::class, SafeTensorsMaterializer.requiredDType(DataType.FLOAT16))
        assertEquals(FP32::class, SafeTensorsMaterializer.requiredDType(DataType.BFLOAT16))
        assertEquals(Int32::class, SafeTensorsMaterializer.requiredDType(DataType.INT32))
        assertEquals(Int32::class, SafeTensorsMaterializer.requiredDType(DataType.INT64))
        assertEquals(Int8::class, SafeTensorsMaterializer.requiredDType(DataType.INT8))
        assertEquals(Int8::class, SafeTensorsMaterializer.requiredDType(DataType.UINT8))
        assertEquals(Int8::class, SafeTensorsMaterializer.requiredDType(DataType.BOOL))
        assertEquals(Int8::class, SafeTensorsMaterializer.requiredDType(DataType.UNKNOWN))
    }
}
