package sk.ainet.data

import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.sliceView
import sk.ainet.lang.types.DType


public data class DataBatch<T : DType, V>(
    val x: Array<Tensor<T, V>>,
    val y: Tensor<T, V>,
    val indices: IntArray = IntArray(y.shape[0]) { it },
    val metadata: Map<String, String> = emptyMap()
) {
    /** Number of samples represented by this batch. */
    public val batchSize: Int get() = indices.size

    /** Returns a copy with [metadata] merged into the existing metadata map. */
    public fun withMetadata(metadata: Map<String, String>): DataBatch<T, V> =
        copy(metadata = this.metadata + metadata)

    /** Returns a contiguous slice of this batch over its leading batch dimension. */
    public fun slice(range: IntRange): DataBatch<T, V> {
        require(!range.isEmpty()) { "range must not be empty" }
        require(range.first >= 0) { "range start must be non-negative" }
        require(range.last < batchSize) { "range end must be within batch bounds" }

        val endExclusive = range.last + 1
        return copy(
            x = x.map { tensor -> tensor.sliceLeadingDimension(range.first, endExclusive) }.toTypedArray(),
            y = y.sliceLeadingDimension(range.first, endExclusive),
            indices = indices.sliceArray(range)
        )
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as DataBatch<*, *>

        if (!x.contentEquals(other.x)) return false
        if (y != other.y) return false
        if (!indices.contentEquals(other.indices)) return false
        if (metadata != other.metadata) return false

        return true
    }

    override fun hashCode(): Int {
        var result = x.contentHashCode()
        result = 31 * result + y.hashCode()
        result = 31 * result + indices.contentHashCode()
        result = 31 * result + metadata.hashCode()
        return result
    }
}

private fun <T : DType, V> Tensor<T, V>.sliceLeadingDimension(start: Int, endExclusive: Int): Tensor<T, V> {
    if (rank == 0) return this
    return sliceView {
        segment { range(start, endExclusive) }
        repeat(rank - 1) {
            segment { all() }
        }
    }
}
