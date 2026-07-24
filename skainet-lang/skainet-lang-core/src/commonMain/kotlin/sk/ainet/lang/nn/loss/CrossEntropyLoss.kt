package sk.ainet.lang.nn.loss

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.logSoftmax
import sk.ainet.lang.tensor.mean
import sk.ainet.lang.tensor.sum
import sk.ainet.lang.tensor.times
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

public class CrossEntropyLoss @kotlin.jvm.JvmOverloads constructor(
    private val dim: Int = -1
) : Loss {

    override fun <T : DType, V> forward(
        preds: Tensor<T, V>,
        targets: Tensor<out DType, *>,
        ctx: ExecutionContext,
        reduction: Reduction
    ): Tensor<T, V> {
        validateFloatPreds(preds)
        val classDim = normalizeDim(dim, preds.rank)
        val classCount = preds.shape[classDim]
        require(classCount > 0) { "CrossEntropyLoss requires class dimension > 0" }

        val perSample = when (targets.dtype) {
            Int32::class -> {
                @Suppress("UNCHECKED_CAST")
                computeIndexTargetLoss(preds, targets as Tensor<Int32, Int>, ctx, classDim, classCount)
            }
            FP32::class, FP16::class -> {
                require(preds.dtype == targets.dtype) {
                    "CrossEntropyLoss requires preds/targets dtype match for soft targets, got ${preds.dtype} vs ${targets.dtype}"
                }
                @Suppress("UNCHECKED_CAST")
                computeSoftTargetLoss(preds, targets as Tensor<T, V>, classDim)
            }
            else -> error("Unsupported target dtype for CrossEntropyLoss: ${targets.dtype}")
        }
        return applyReduction(perSample, reduction)
    }

    private fun <T : DType, V> computeIndexTargetLoss(
        preds: Tensor<T, V>,
        targets: Tensor<Int32, Int>,
        ctx: ExecutionContext,
        classDim: Int,
        classCount: Int
    ): Tensor<T, V> {
        require(targets.rank == preds.rank - 1) {
            "CrossEntropyLoss expected target rank ${preds.rank - 1} for class indices, got ${targets.rank}"
        }
        validateIndexTargetShapes(preds, targets, classDim)

        val logProbs = preds.logSoftmax(classDim)

        // Build a one-hot selector for the target classes as a constant tensor,
        // then compute the NLL with differentiable ops: -sum_c(oneHot * logProbs).
        // Reading the class indices host-side to construct the (non-differentiable)
        // one-hot is fine; the gradient path stays intact because the multiply and
        // sum are recorded through the predictions' ops — unlike the previous
        // host-side result construction, which detached the tape.
        val oneHotData = ctx.tensorDataFactory.init<T, V>(preds.shape, preds.dtype) { idx ->
            val sampleIdx = removeClassIndex(idx, classDim)
            val cls = targets.data.get(*sampleIdx) as Int
            require(cls in 0 until classCount) {
                "CrossEntropyLoss target index $cls out of range [0, $classCount)"
            }
            @Suppress("UNCHECKED_CAST")
            (if (idx[classDim] == cls) 1f else 0f) as V
        }
        val oneHot = ctx.fromData(oneHotData, preds.dtype)
        val weighted = logProbs.ops.multiply(logProbs, oneHot)
        val perSample = weighted.ops.sum(weighted, classDim)
        return perSample.ops.mulScalar(perSample, -1.0)
    }

    private fun <T : DType, V> computeSoftTargetLoss(
        preds: Tensor<T, V>,
        targets: Tensor<T, V>,
        classDim: Int
    ): Tensor<T, V> {
        validateFloatTargets(targets)
        require(preds.shape == targets.shape) {
            "CrossEntropyLoss with soft targets requires preds/targets shape match, got ${preds.shape.dimensions.contentToString()} vs ${targets.shape.dimensions.contentToString()}"
        }
        val logProbs = preds.logSoftmax(classDim)
        // Dispatch the multiply through the predictions' ops (not `targets * ...`,
        // which would route through targets.ops and detach the tape when the
        // targets live on a non-recording context).
        val weighted = logProbs.ops.multiply(logProbs, targets)
        val summed = weighted.ops.sum(weighted, classDim)
        return summed.ops.mulScalar(summed, -1.0)
    }

    /** Drops the class-dimension coordinate from a full prediction index. */
    private fun removeClassIndex(fullIdx: IntArray, classDim: Int): IntArray {
        val result = IntArray(fullIdx.size - 1)
        var j = 0
        for (i in fullIdx.indices) if (i != classDim) result[j++] = fullIdx[i]
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
                "CrossEntropyLoss target shape mismatch at dim $pi: expected ${predDims[pi]}, got ${targetDims[ti]}"
            }
            ti++
        }
    }

    private fun normalizeDim(dim: Int, rank: Int): Int {
        val nd = if (dim < 0) dim + rank else dim
        require(nd in 0 until rank) { "Dimension $dim out of range for rank $rank" }
        return nd
    }

    private fun <T : DType, V> applyReduction(
        loss: Tensor<T, V>,
        reduction: Reduction
    ): Tensor<T, V> = when (reduction) {
        Reduction.NONE -> loss
        Reduction.SUM -> loss.sum()
        Reduction.MEAN -> loss.mean()
    }

    private fun <T : DType, V> validateFloatPreds(preds: Tensor<T, V>) {
        require(preds.dtype == FP32::class || preds.dtype == FP16::class) {
            "CrossEntropyLoss requires floating point predictions, got ${preds.dtype}"
        }
    }

    private fun <T : DType, V> validateFloatTargets(targets: Tensor<T, V>) {
        require(targets.dtype == FP32::class || targets.dtype == FP16::class) {
            "CrossEntropyLoss soft targets must be floating point, got ${targets.dtype}"
        }
    }
}
