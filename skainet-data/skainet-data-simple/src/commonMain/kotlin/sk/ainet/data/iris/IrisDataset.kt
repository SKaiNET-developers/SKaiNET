package sk.ainet.data.iris

import sk.ainet.context.DefaultDataExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.data.DataBatch
import sk.ainet.data.Dataset
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import kotlin.math.min
import kotlin.random.Random

/**
 * The Iris dataset (Fisher, 1936) as a [Dataset]: 150 samples, 4 features,
 * 3 balanced classes.
 *
 * Batching produces FP32 tensors:
 *   x -> Tensor<FP32, Float> [batch, 4]  raw measurements in cm (row-major,
 *       ordered by [Iris.featureNames])
 *   y -> Tensor<FP32, Float> [batch, 3]  one-hot species vectors
 *
 * [getY] deliberately returns the class index ([Int]) rather than the one-hot
 * vector: stratified splitting buckets samples by `Y`, which requires value
 * equality. The one-hot conversion happens inside batch construction.
 */
public data class IrisDataset(
    val samples: List<IrisSample>,
    private val executionContext: ExecutionContext = DefaultDataExecutionContext()
) : Dataset<FloatArray, Int>() {

    override val inputShape: Shape get() = Shape(Iris.featureNames.size)

    override val outputShape: Shape get() = Shape(Iris.classNames.size)

    override val xSize: Int get() = samples.size

    /** Returns the four measurements of sample [idx], ordered by [Iris.featureNames]. */
    override fun getX(idx: Int): FloatArray = samples[idx].toFeatures()

    /** Returns the species class index of sample [idx]. */
    override fun getY(idx: Int): Int = samples[idx].label

    override fun shuffle(): Dataset<FloatArray, Int> =
        IrisDataset(samples.shuffled(Random.Default), executionContext)

    override fun split(splitRatio: Double): Pair<Dataset<FloatArray, Int>, Dataset<FloatArray, Int>> {
        require(splitRatio > 0.0 && splitRatio < 1.0) { "splitRatio must be in (0,1)" }
        val at = (samples.size * splitRatio).toInt()
        return IrisDataset(samples.subList(0, at).toList(), executionContext) to
            IrisDataset(samples.subList(at, samples.size).toList(), executionContext)
    }

    /**
     * Creates a data batch over the contiguous range starting at [batchStart].
     *
     * Delegates to [createBatchFor]; see also [createIndexedDataBatch], which
     * serves shuffled, split and filtered dataset views.
     */
    override fun <T : DType, V> createDataBatch(batchStart: Int, batchLength: Int): DataBatch<T, V> {
        val length = min(batchLength, xSize - batchStart)
        return createBatchFor(IntArray(length) { offset -> batchStart + offset })
    }

    /**
     * Creates a data batch for arbitrary logical sample [indices].
     *
     * This override is what keeps `split(...)`, `shuffle(...)` and `filter { }`
     * views working with tensor batching: those views hold non-contiguous
     * indices and would otherwise hit the base class' contiguous-only default.
     */
    override fun <T : DType, V> createIndexedDataBatch(indices: IntArray): DataBatch<T, V> =
        createBatchFor(indices)

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> createBatchFor(indices: IntArray): DataBatch<T, V> {
        require(indices.isNotEmpty()) { "indices must not be empty" }
        val n = indices.size
        val featureCount = Iris.featureNames.size
        val classCount = Iris.classNames.size

        val xData = FloatArray(n * featureCount)
        val yData = FloatArray(n * classCount)
        indices.forEachIndexed { row, sampleIndex ->
            val sample = samples[sampleIndex]
            val features = sample.toFeatures()
            features.copyInto(xData, destinationOffset = row * featureCount)
            yData[row * classCount + sample.label] = 1.0f
        }

        val x: Tensor<FP32, Float> =
            executionContext.fromFloatArray(Shape(n, featureCount), FP32::class, xData)
        val y: Tensor<FP32, Float> =
            executionContext.fromFloatArray(Shape(n, classCount), FP32::class, yData)

        return DataBatch(
            x = arrayOf(x) as Array<Tensor<T, V>>,
            y = y as Tensor<T, V>,
            indices = indices.copyOf()
        )
    }

    /** Returns a subset of the dataset covering `[fromIndex, toIndex)`. */
    public fun subset(fromIndex: Int, toIndex: Int): IrisDataset =
        IrisDataset(samples.subList(fromIndex, toIndex), executionContext)
}
