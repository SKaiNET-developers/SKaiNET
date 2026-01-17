package sk.ainet.lang.nn.metrics

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

/**
 * Classification accuracy metric.
 *
 * Computes the fraction of predictions that match the target labels.
 * Supports both hard targets (class indices) and soft targets (one-hot or probabilities).
 *
 * For predictions, the class with maximum value along [dim] is selected.
 * For soft targets, the class with maximum value is used as the ground truth.
 * For hard targets (Int32), the value directly represents the class index.
 *
 * @param dim The dimension along which to find the predicted class (default: -1, last dimension)
 * @param threshold Optional threshold for binary classification. If provided, predictions > threshold
 *                  are classified as class 1, otherwise class 0. Only applicable when dim size is 1 or 2.
 */
public class Accuracy(
    private val dim: Int = -1,
    private val threshold: Float? = null
) : Metric {

    override val name: String = "accuracy"

    private var correctCount: Long = 0
    private var totalCount: Long = 0

    override fun <T : DType, V> update(
        predictions: Tensor<T, V>,
        targets: Tensor<out DType, *>,
        ctx: ExecutionContext
    ) {
        validatePredictions(predictions)
        val classDim = normalizeDim(dim, predictions.rank)

        val (batchCorrect, batchTotal) = when (targets.dtype) {
            Int32::class -> {
                @Suppress("UNCHECKED_CAST")
                computeWithIndexTargets(predictions, targets as Tensor<Int32, Int>, classDim)
            }
            FP32::class, FP16::class -> {
                @Suppress("UNCHECKED_CAST")
                computeWithSoftTargets(predictions, targets as Tensor<out DType, *>, classDim)
            }
            else -> error("Unsupported target dtype for Accuracy: ${targets.dtype}")
        }

        correctCount += batchCorrect
        totalCount += batchTotal
    }

    override fun compute(): Double {
        if (totalCount == 0L) return 0.0
        return correctCount.toDouble() / totalCount.toDouble()
    }

    override fun reset() {
        correctCount = 0
        totalCount = 0
    }

    private fun <T : DType, V> computeWithIndexTargets(
        predictions: Tensor<T, V>,
        targets: Tensor<Int32, Int>,
        classDim: Int
    ): Pair<Long, Long> {
        // Validate shapes: targets should have one less dimension than predictions
        // (the class dimension is removed)
        require(targets.rank == predictions.rank - 1) {
            "Accuracy expected target rank ${predictions.rank - 1} for class indices, got ${targets.rank}"
        }
        validateIndexTargetShapes(predictions, targets, classDim)

        val classCount = predictions.shape[classDim]
        var correct = 0L
        var total = 0L

        // Iterate over all samples
        iterateOverSamples(targets.shape.dimensions) { sampleIdx ->
            val targetClass = targets.data.get(*sampleIdx)
            val predictedClass = if (threshold != null && classCount <= 2) {
                // Binary classification with threshold
                val logitIdx = insertClassIndex(sampleIdx, if (classCount == 2) 1 else 0, classDim, predictions.rank)
                val logit = predictions.data.get(*logitIdx) as Float
                if (logit > threshold) 1 else 0
            } else {
                // Multi-class: argmax along class dimension
                argmax(predictions, sampleIdx, classDim)
            }

            if (predictedClass == targetClass) {
                correct++
            }
            total++
        }

        return correct to total
    }

    private fun <T : DType, V> computeWithSoftTargets(
        predictions: Tensor<T, V>,
        targets: Tensor<out DType, *>,
        classDim: Int
    ): Pair<Long, Long> {
        // Soft targets should have the same shape as predictions
        require(predictions.shape == targets.shape) {
            "Accuracy with soft targets requires matching shapes, got ${predictions.shape.dimensions.contentToString()} vs ${targets.shape.dimensions.contentToString()}"
        }

        val classCount = predictions.shape[classDim]
        var correct = 0L
        var total = 0L

        // Get shape without the class dimension for iteration
        val sampleShape = predictions.shape.dimensions.filterIndexed { i, _ -> i != classDim }.toIntArray()

        iterateOverSamples(sampleShape) { sampleIdx ->
            val predictedClass = if (threshold != null && classCount <= 2) {
                val logitIdx = insertClassIndex(sampleIdx, if (classCount == 2) 1 else 0, classDim, predictions.rank)
                val logit = predictions.data.get(*logitIdx) as Float
                if (logit > threshold) 1 else 0
            } else {
                argmax(predictions, sampleIdx, classDim)
            }

            val targetClass = argmaxTarget(targets, sampleIdx, classDim)

            if (predictedClass == targetClass) {
                correct++
            }
            total++
        }

        return correct to total
    }

    private fun <T : DType, V> argmax(
        tensor: Tensor<T, V>,
        sampleIdx: IntArray,
        classDim: Int
    ): Int {
        val classCount = tensor.shape[classDim]
        var maxIdx = 0
        var maxVal = Float.NEGATIVE_INFINITY

        for (c in 0 until classCount) {
            val fullIdx = insertClassIndex(sampleIdx, c, classDim, tensor.rank)
            val value = tensor.data.get(*fullIdx) as Float
            if (value > maxVal) {
                maxVal = value
                maxIdx = c
            }
        }
        return maxIdx
    }

    private fun argmaxTarget(
        tensor: Tensor<out DType, *>,
        sampleIdx: IntArray,
        classDim: Int
    ): Int {
        val classCount = tensor.shape[classDim]
        var maxIdx = 0
        var maxVal = Float.NEGATIVE_INFINITY

        for (c in 0 until classCount) {
            val fullIdx = insertClassIndex(sampleIdx, c, classDim, tensor.rank)
            val value = when (tensor.dtype) {
                FP32::class -> tensor.data.get(*fullIdx) as Float
                FP16::class -> (tensor.data.get(*fullIdx) as Number).toFloat()
                else -> error("Unsupported dtype")
            }
            if (value > maxVal) {
                maxVal = value
                maxIdx = c
            }
        }
        return maxIdx
    }

    private fun insertClassIndex(
        baseIdx: IntArray,
        cls: Int,
        classDim: Int,
        outRank: Int
    ): IntArray {
        val result = IntArray(outRank)
        var bi = 0
        for (i in 0 until outRank) {
            if (i == classDim) {
                result[i] = cls
            } else {
                result[i] = baseIdx[bi++]
            }
        }
        return result
    }

    private fun validateIndexTargetShapes(
        preds: Tensor<*, *>,
        targets: Tensor<*, *>,
        classDim: Int
    ) {
        val predDims = preds.shape.dimensions
        val targetDims = targets.shape.dimensions
        var ti = 0
        for (pi in predDims.indices) {
            if (pi == classDim) continue
            require(predDims[pi] == targetDims[ti]) {
                "Accuracy target shape mismatch at dim $pi: expected ${predDims[pi]}, got ${targetDims[ti]}"
            }
            ti++
        }
    }

    private fun <T : DType, V> validatePredictions(preds: Tensor<T, V>) {
        require(preds.dtype == FP32::class || preds.dtype == FP16::class) {
            "Accuracy requires floating point predictions, got ${preds.dtype}"
        }
    }

    private fun normalizeDim(dim: Int, rank: Int): Int {
        val nd = if (dim < 0) dim + rank else dim
        require(nd in 0 until rank) { "Dimension $dim out of range for rank $rank" }
        return nd
    }

    private inline fun iterateOverSamples(shape: IntArray, action: (IntArray) -> Unit) {
        if (shape.isEmpty()) {
            action(IntArray(0))
            return
        }

        val idx = IntArray(shape.size)
        val total = shape.fold(1) { acc, d -> acc * d }

        repeat(total) {
            action(idx)
            // Increment index
            for (d in shape.lastIndex downTo 0) {
                idx[d]++
                if (idx[d] < shape[d]) break
                idx[d] = 0
            }
        }
    }
}

/**
 * Factory function for Accuracy metric.
 *
 * @param dim The dimension along which to find the predicted class (default: -1)
 * @param threshold Optional threshold for binary classification
 */
public fun accuracy(dim: Int = -1, threshold: Float? = null): Metric = Accuracy(dim, threshold)

/**
 * Binary accuracy metric with a threshold of 0.5.
 * Useful for binary classification problems with sigmoid outputs.
 */
public fun binaryAccuracy(threshold: Float = 0.5f): Metric = Accuracy(dim = -1, threshold = threshold)
