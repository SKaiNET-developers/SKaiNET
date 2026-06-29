package sk.ainet.data

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType
import kotlin.math.min
import kotlin.random.Random


/** Just abstract Dataset. */
public abstract class Dataset<T, Y> {
    /** Optional shape metadata for one input sample. */
    public open val inputShape: Shape? get() = null

    /** Optional shape metadata for one output sample. */
    public open val outputShape: Shape? get() = null

    /** Splits datasets on two sub-datasets according [splitRatio].*/
    public abstract fun split(splitRatio: Double): Pair<Dataset<T, Y>, Dataset<T, Y>>

    /** Returns amount of data rows. */
    public abstract val xSize: Int

    /** Returns amount of data rows. Alias for [xSize]. */
    public val size: Int get() = xSize

    /** Returns row by index [idx]. */
    public abstract fun getX(idx: Int): T

    /** Returns label as [Int] by index [idx]. */
    public abstract fun getY(idx: Int): Y

    /** Shuffles the dataset. */
    public abstract fun shuffle(): Dataset<T, Y>

    /** Shuffles the dataset, using [seed] when deterministic ordering is required. */
    public open fun shuffle(seed: Long? = null): Dataset<T, Y> {
        if (seed == null) return shuffle()
        val indices = IntArray(xSize) { it }
        indices.shuffle(Random(seed))
        return IndexedDataset(this, indices)
    }

    /**
     * Splits the dataset with optional deterministic shuffling and label stratification.
     *
     * The original [split] remains the compatibility path. This overload adds the
     * ML-oriented behavior expected by training pipelines without forcing every
     * concrete dataset to duplicate the same index bookkeeping.
     */
    public open fun split(
        splitRatio: Double,
        seed: Long? = null,
        stratified: Boolean = false
    ): Pair<Dataset<T, Y>, Dataset<T, Y>> {
        require(splitRatio > 0.0 && splitRatio < 1.0) { "splitRatio must be in (0,1)" }
        if (seed == null && !stratified) return split(splitRatio)

        val (leftIndices, rightIndices) = if (stratified) {
            stratifiedSplitIndices(splitRatio, seed)
        } else {
            val indices = IntArray(xSize) { it }
            seed?.let { indices.shuffle(Random(it)) }
            indices.splitAtRatio(splitRatio)
        }

        return IndexedDataset(this, leftIndices) to IndexedDataset(this, rightIndices)
    }

    /** Returns a dataset view containing only samples accepted by [predicate]. */
    public fun filter(predicate: (T, Y) -> Boolean): Dataset<T, Y> {
        val indices = (0 until xSize)
            .filter { idx -> predicate(getX(idx), getY(idx)) }
            .toIntArray()
        return IndexedDataset(this, indices)
    }

    /** Returns a dataset view with transformed input samples. */
    public fun <NX> mapX(transform: (T) -> NX): Dataset<NX, Y> =
        MappedDataset(this) { x, y -> transform(x) to y }

    /** Returns a dataset view with transformed target samples. */
    public fun <NY> mapY(transform: (Y) -> NY): Dataset<T, NY> =
        MappedDataset(this) { x, y -> x to transform(y) }

    /** Returns a dataset view with transformed input and target samples. */
    public fun <NX, NY> transform(transformer: (T, Y) -> Pair<NX, NY>): Dataset<NX, NY> =
        MappedDataset(this, transformer)

    /**
     * An iterator over a [Dataset].
     */
    public inner class BatchIterator<T : DType, V> internal constructor(
        private val batchSize: Int
    ) : Iterator<DataBatch<T, V>> {

        private var batchStart = 0

        override fun hasNext(): Boolean = batchStart < xSize

        override fun next(): DataBatch<T, V> {
            val batchLength = min(batchSize, xSize - batchStart)
            val batch = createDataBatch<T, V>(batchStart, batchLength)
            batchStart += batchSize
            return batch
        }
    }

    /** Creates data batch that starts from [batchStart] with length [batchLength]. */
    protected abstract fun <T : DType, V> createDataBatch(batchStart: Int, batchLength: Int): DataBatch<T, V>

    /**
     * Creates a data batch for arbitrary logical sample [indices].
     *
     * Concrete datasets that can tensorize non-contiguous samples should override
     * this method. The default path supports contiguous ranges and fails fast for
     * non-contiguous index views instead of silently returning the wrong rows.
     */
    protected open fun <T : DType, V> createIndexedDataBatch(indices: IntArray): DataBatch<T, V> {
        require(indices.isNotEmpty()) { "indices must not be empty" }
        val first = indices.first()
        require(first >= 0) { "indices must be non-negative" }
        val contiguous = indices.withIndex().all { (offset, value) -> value == first + offset }
        require(contiguous) {
            "Non-contiguous data batches require createIndexedDataBatch(indices) support in the concrete dataset"
        }
        return createDataBatch(first, indices.size)
    }

    /** Creates data batch that starts from [batchStart] with length [batchLength]. */
    public fun <T : DType, V> dataBatch(batchStart: Int, batchLength: Int): DataBatch<T, V> {
        require(batchStart >= 0) { "batchStart must be non-negative" }
        require(batchLength >= 0) { "batchLength must be non-negative" }
        require(batchStart + batchLength <= xSize) { "batch exceeds dataset size" }
        return createDataBatch(batchStart, batchLength)
    }

    /** Creates a data batch for arbitrary logical sample [indices]. */
    public fun <T : DType, V> dataBatch(indices: IntArray): DataBatch<T, V> {
        require(indices.all { it in 0 until xSize }) { "indices must be inside dataset bounds" }
        return createIndexedDataBatch(indices)
    }

    /** Returns [BatchIterator] with fixed [batchSize]. */
    public fun <T : DType, V> batchIterator(batchSize: Int): BatchIterator<T, V> {
        require(batchSize > 0) { "batchSize must be positive" }
        return BatchIterator(batchSize)
    }

    /** Returns a cold [Flow] of data batches. */
    public fun <T : DType, V> batches(
        batchSize: Int,
        shuffle: Boolean = true,
        seed: Long? = null
    ): Flow<DataBatch<T, V>> = flow {
        val source = if (shuffle) shuffle(seed) else this@Dataset
        val iterator = source.batchIterator<T, V>(batchSize)
        while (iterator.hasNext()) {
            emit(iterator.next())
        }
    }

    /** Returns a cold [Flow] over [epochCount] epochs of data batches. */
    public fun <T : DType, V> epochs(
        epochCount: Int,
        batchSize: Int,
        shuffle: Boolean = true,
        seed: Long? = null
    ): Flow<DataBatch<T, V>> = flow {
        require(epochCount >= 0) { "epochCount must be non-negative" }
        for (epoch in 0 until epochCount) {
            val epochSeed = seed?.plus(epoch)
            val iterator = (if (shuffle) shuffle(epochSeed) else this@Dataset).batchIterator<T, V>(batchSize)
            while (iterator.hasNext()) {
                emit(iterator.next())
            }
        }
    }

    private fun stratifiedSplitIndices(splitRatio: Double, seed: Long?): Pair<IntArray, IntArray> {
        val buckets = LinkedHashMap<Y, MutableList<Int>>()
        for (idx in 0 until xSize) {
            buckets.getOrPut(getY(idx)) { mutableListOf() }.add(idx)
        }

        val random = seed?.let { Random(it) }
        val left = mutableListOf<Int>()
        val right = mutableListOf<Int>()

        for (bucket in buckets.values) {
            val indices = bucket.toMutableList()
            if (random != null) indices.shuffle(random)
            val splitIndex = (indices.size * splitRatio).toInt().coerceIn(0, indices.size)
            left.addAll(indices.subList(0, splitIndex))
            right.addAll(indices.subList(splitIndex, indices.size))
        }

        if (random != null) {
            left.shuffle(random)
            right.shuffle(random)
        }

        return left.toIntArray() to right.toIntArray()
    }
}

private class IndexedDataset<X, Y>(
    private val source: Dataset<X, Y>,
    private val indices: IntArray
) : Dataset<X, Y>() {
    override val inputShape: Shape? get() = source.inputShape
    override val outputShape: Shape? get() = source.outputShape
    override val xSize: Int get() = indices.size

    override fun getX(idx: Int): X = source.getX(indices[idx])

    override fun getY(idx: Int): Y = source.getY(indices[idx])

    override fun shuffle(): Dataset<X, Y> {
        val shuffled = indices.copyOf()
        shuffled.shuffle(Random.Default)
        return IndexedDataset(source, shuffled)
    }

    override fun split(splitRatio: Double): Pair<Dataset<X, Y>, Dataset<X, Y>> {
        require(splitRatio > 0.0 && splitRatio < 1.0) { "splitRatio must be in (0,1)" }
        val (left, right) = indices.splitAtRatio(splitRatio)
        return IndexedDataset(source, left) to IndexedDataset(source, right)
    }

    override fun <T : DType, V> createDataBatch(batchStart: Int, batchLength: Int): DataBatch<T, V> {
        val actualLength = min(batchLength, xSize - batchStart)
        val batchIndices = IntArray(actualLength) { offset -> indices[batchStart + offset] }
        return source.dataBatch(batchIndices)
    }

    override fun <T : DType, V> createIndexedDataBatch(indices: IntArray): DataBatch<T, V> {
        val sourceIndices = IntArray(indices.size) { offset -> this.indices[indices[offset]] }
        return source.dataBatch(sourceIndices)
    }
}

private class MappedDataset<SX, SY, TX, TY>(
    private val source: Dataset<SX, SY>,
    private val transformer: (SX, SY) -> Pair<TX, TY>
) : Dataset<TX, TY>() {
    override val xSize: Int get() = source.xSize

    override fun getX(idx: Int): TX {
        val (x, _) = transformer(source.getX(idx), source.getY(idx))
        return x
    }

    override fun getY(idx: Int): TY {
        val (_, y) = transformer(source.getX(idx), source.getY(idx))
        return y
    }

    override fun shuffle(): Dataset<TX, TY> = shuffle(Random.nextLong())

    override fun split(splitRatio: Double): Pair<Dataset<TX, TY>, Dataset<TX, TY>> =
        split(splitRatio, seed = null, stratified = false)

    override fun <T : DType, V> createDataBatch(batchStart: Int, batchLength: Int): DataBatch<T, V> {
        throw UnsupportedOperationException("MappedDataset cannot create tensor batches without a tensorization transform")
    }
}

private fun IntArray.splitAtRatio(splitRatio: Double): Pair<IntArray, IntArray> {
    val splitIndex = (size * splitRatio).toInt().coerceIn(0, size)
    return copyOfRange(0, splitIndex) to copyOfRange(splitIndex, size)
}
