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
        val outData = ctx.tensorDataFactory.init<T, V>(targets.shape, preds.dtype) { idx ->
            val cls = targets.data.get(*idx) as Int
            require(cls in 0 until classCount) {
                "CrossEntropyLoss target index $cls out of range [0, $classCount)"
            }
            val logIdx = insertClassIndex(idx, cls, classDim, preds.rank)
            val logVal = logProbs.data.get(*logIdx) as Float
            @Suppress("UNCHECKED_CAST")
            (-logVal) as V
        }
        return ctx.fromData(outData, preds.dtype)
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
        val weighted = targets * logProbs
        val summed = weighted.sum(classDim)
        return (-1f) * summed
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
