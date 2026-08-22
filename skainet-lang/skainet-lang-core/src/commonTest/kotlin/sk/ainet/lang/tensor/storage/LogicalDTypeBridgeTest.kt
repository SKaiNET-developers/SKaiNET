package sk.ainet.lang.tensor.storage

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.FP64
import sk.ainet.lang.types.Int16
import sk.ainet.lang.types.Int32
import sk.ainet.lang.types.Int4
import sk.ainet.lang.types.Int64
import sk.ainet.lang.types.Int8
import sk.ainet.lang.types.Ternary
import sk.ainet.lang.types.UInt16
import sk.ainet.lang.types.UInt32
import sk.ainet.lang.types.UInt64
import sk.ainet.lang.types.UInt8
import sk.ainet.lang.types.isFloatingPoint
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame

/**
 * SKEEP-003 Phase 0, decision #13: the two-way `LogicalDType` <-> `DType` bridge must be total
 * and bijective so that `LogicalDType` can later be merged into `DType` without a semantic gap.
 */
class LogicalDTypeBridgeTest {

    private val expectedPairs: List<Pair<LogicalDType, DType>> = listOf(
        LogicalDType.TERNARY to Ternary,
        LogicalDType.INT4 to Int4,
        LogicalDType.INT8 to Int8,
        LogicalDType.INT16 to Int16,
        LogicalDType.INT32 to Int32,
        LogicalDType.INT64 to Int64,
        LogicalDType.UINT8 to UInt8,
        LogicalDType.UINT16 to UInt16,
        LogicalDType.UINT32 to UInt32,
        LogicalDType.UINT64 to UInt64,
        LogicalDType.FLOAT16 to FP16,
        LogicalDType.BFLOAT16 to BF16,
        LogicalDType.FLOAT32 to FP32,
        LogicalDType.FLOAT64 to FP64,
    )

    @Test
    fun bothTypeSystemsHaveFourteenMembers() {
        assertEquals(14, LogicalDType.entries.size)
        assertEquals(14, DType.getAllTypes().size)
        assertEquals(14, expectedPairs.size)
    }

    @Test
    fun explicitPairTable() {
        for ((logical, dtype) in expectedPairs) {
            assertSame(dtype, logical.toDType(), "toDType of $logical")
            assertEquals(logical, dtype.toLogicalDType(), "toLogicalDType of ${dtype.name}")
        }
    }

    @Test
    fun logicalToDTypeAndBackIsIdentity() {
        for (logical in LogicalDType.entries) {
            assertEquals(logical, logical.toDType().toLogicalDType(), "round trip of $logical")
        }
    }

    @Test
    fun dtypeToLogicalAndBackIsIdentity() {
        for (dtype in DType.getAllTypes().values) {
            assertSame(dtype, dtype.toLogicalDType().toDType(), "round trip of ${dtype.name}")
        }
    }

    @Test
    fun mappingIsBijective() {
        val images = LogicalDType.entries.map { it.toDType() }.toSet()
        assertEquals(LogicalDType.entries.size, images.size, "toDType must be injective")
        val preImages = DType.getAllTypes().values.map { it.toLogicalDType() }.toSet()
        assertEquals(DType.getAllTypes().size, preImages.size, "toLogicalDType must be injective")
    }

    @Test
    fun widthAndFloatnessAgreeAcrossTheBridge() {
        for (logical in LogicalDType.entries) {
            val dtype = logical.toDType()
            assertEquals(logical.sizeInBits, dtype.sizeInBits, "sizeInBits of $logical")
            assertEquals(logical.isFloatingPoint, dtype.isFloatingPoint(), "isFloatingPoint of $logical")
        }
    }

    @Test
    fun descriptorsExposeTheDType() {
        assertSame(FP16, StorageSpec.fromDType(FP16).dtype)
        assertSame(FP32, StorageSpec.q4k().dtype)
        assertSame(FP32, StorageSpec.q80().dtype)
        assertSame(BF16, StorageSpec.borrowed(BF16).dtype)
        assertSame(Int8, StorageSpec.mmapWeights(Int8).dtype)

        val storage = TensorStorage(
            shape = Shape(2, 3),
            logicalType = LogicalDType.FLOAT32,
            encoding = TensorEncoding.Dense(4),
            buffer = BufferHandle.Owned(ByteArray(24)),
        )
        assertSame(FP32, storage.dtype)
        assertSame(FP32, storage.memoryReport().dtype)
    }
}
