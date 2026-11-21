package sk.ainet.lang.tensor

import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.jvm.JvmInline

/** Thrown when an index tensor contains a non-integral value while strict validation is enabled. */
public class NonIntegralIndexException(message: String) : IllegalArgumentException(message)

/** Thrown when an index value is outside the allowed range. */
public class IndexOutOfRangeException(message: String) : IllegalArgumentException(message)

/**
 * Lightweight wrapper marking a Tensor as index-bearing. It does not change storage,
 * only annotates intent and guarantees validation performed by [asIndices].
 */
@JvmInline
public value class IndexTensor<V>(public val t: Tensor<*, V>)

/** Returns true if the provided dtype is considered an integer storage type. */
private fun DType.isIntegerType(): Boolean = this is Int32

/** Returns true if the provided dtype is considered a float storage type. */
private fun DType.isFloatType(): Boolean = this is FP32 || this is FP16

/**
 * Validates this tensor as an index tensor and wraps it into [IndexTensor].
 * - For integer storage dtypes, this is a no-op wrap.
 * - For float storage dtypes, checks all elements are mathematically integral when [strict] is true.
 */
public fun <T : DType, V> Tensor<T, V>.asIndices(strict: Boolean = true): IndexTensor<V> {
    val dt = this.dtype
    val dtypeInstance: DType = when (val c = dt) {
        Int32::class -> Int32
        FP32::class -> FP32
        FP16::class -> FP16
        else -> FP32 // default best-effort; unknown classes treat as float
    }

    if (dtypeInstance.isIntegerType()) return IndexTensor(this)

    if (strict && dtypeInstance.isFloatType()) {
        // fast path for direct FloatArray-backed tensor data
        val data = this.data
        val heapFloats = data.heapFloats
        when {
            heapFloats != null -> {
                val buf = heapFloats
                val start = data.heapFloatsOffset
                for (i in 0 until this.volume) {
                    val f = buf[start + i]
                    val fi = f.toInt()
                    if (fi.toFloat() != f) {
                        throw NonIntegralIndexException("Non-integral index value at flat position $i: $f. Use integer tensors or call asIndices(strict=false) if intentional.")
                    }
                }
            }
            else -> {
                // generic path via ItemsAccessor
                val vol = this.volume
                // iterate by linear index using 1D indexing on data with [i]
                for (i in 0 until vol) {
                    val any = data[i]
                    if (any is Number) {
                        val f = any.toFloat()
                        val fi = f.toInt()
                        if (fi.toFloat() != f) {
                            throw NonIntegralIndexException("Non-integral index value at flat position $i: $f. Use integer tensors or call asIndices(strict=false) if intentional.")
                        }
                    }
                }
            }
        }
    }
    return IndexTensor(this)
}
