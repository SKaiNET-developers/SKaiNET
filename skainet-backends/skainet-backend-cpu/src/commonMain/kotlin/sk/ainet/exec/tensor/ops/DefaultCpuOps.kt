package sk.ainet.exec.tensor.ops

import sk.ainet.lang.tensor.GradState
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType
import sk.ainet.lang.ops.TensorOp
import sk.ainet.lang.ops.InProgress
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.tensor.ops.UpsampleMode
import sk.ainet.lang.types.FP32
import kotlin.math.sqrt

@InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#defaultcpuops")
public open class DefaultCpuOpsBase(protected val dataFactory: TensorDataFactory) : TensorOps {

    protected class CpuTensor<T : DType, V>(
        override val data: sk.ainet.lang.tensor.data.TensorData<T, V>,
        private val opsRef: TensorOps,
        override val dtype: kotlin.reflect.KClass<T>,
        override val gradState: GradState<T, V> = GradState()
    ) : Tensor<T, V> {
        override val ops: TensorOps
            get() = opsRef
    }

    protected fun <T : DType, V> gradStateFrom(vararg tensors: Tensor<T, V>): GradState<T, V> {
        val requires = tensors.any { it.requiresGrad }
        return GradState(requiresGrad = requires)
    }

    protected fun <T : DType, V> newTensor(
        data: sk.ainet.lang.tensor.data.TensorData<T, V>,
        dtype: kotlin.reflect.KClass<T>,
        vararg inputs: Tensor<T, V>
    ): Tensor<T, V> = CpuTensor(data, this, dtype, gradStateFrom(*inputs))

    protected fun broadcastShapes(a: Shape, b: Shape): Shape {
        val ad = a.dimensions
        val bd = b.dimensions
        val maxRank = maxOf(ad.size, bd.size)
        val out = IntArray(maxRank)
        var ai = ad.size - 1
        var bi = bd.size - 1
        for (oi in maxRank - 1 downTo 0) {
            val asz = if (ai >= 0) ad[ai] else 1
            val bsz = if (bi >= 0) bd[bi] else 1
            if (asz != bsz && asz != 1 && bsz != 1) {
                throw IllegalArgumentException("Shapes ${a.dimensions.contentToString()} and ${b.dimensions.contentToString()} cannot be broadcasted")
            }
            out[oi] = maxOf(asz, bsz)
            ai--; bi--
        }
        return Shape(out)
    }

    protected fun mapIndex(idx: IntArray, inShape: Shape): IntArray {
        // Map output index to input index with broadcasting: if input dim == 1, use 0 for that dim.
        val inDims = inShape.dimensions
        val outRank = idx.size
        val inRank = inDims.size
        val mapped = IntArray(inRank)
        var ir = inRank - 1
        var or = outRank - 1
        while (ir >= 0) {
            val inDim = inDims[ir]
            val outIndex = if (or >= 0) idx[or] else 0
            mapped[ir] = if (inDim == 1) 0 else outIndex
            ir--; or--
        }
        return mapped
    }

    protected fun <T : DType, V> requireSameDType(a: Tensor<T, V>, b: Tensor<T, V>) {
        require(a.dtype == b.dtype) { "DType mismatch: ${'$'}{a.dtype} vs ${'$'}{b.dtype}" }
    }

    protected fun <T : DType, V> elementwise(
        a: Tensor<T, V>,
        b: Tensor<T, V>,
        op: (av: V, bv: V, dtype: kotlin.reflect.KClass<T>) -> V
    ): Tensor<T, V> {
        requireSameDType(a, b)
        val outShape = broadcastShapes(a.shape, b.shape)
        val outData = dataFactory.init<T, V>(outShape, a.dtype) { outIdx ->
            val ai = mapIndex(outIdx, a.shape)
            val bi = mapIndex(outIdx, b.shape)
            val av = a.data.get(*ai)
            val bv = b.data.get(*bi)
            op(av, bv, a.dtype)
        }
        return newTensor(outData, a.dtype, a, b)
    }

    // Scalar ops implemented via materializing a full-like tensor and delegating to elementwise ops
    override fun <T : DType, V> addScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val sb = newTensor(
            dataFactory.full<T, V>(a.shape, a.dtype, b),
            a.dtype,
            a
        )
        return add(a, sb)
    }

    override fun <T : DType, V> subScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val sb = newTensor(
            dataFactory.full<T, V>(a.shape, a.dtype, b),
            a.dtype,
            a
        )
        return subtract(a, sb)
    }

    override fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val sb = newTensor(
            dataFactory.full<T, V>(a.shape, a.dtype, b),
            a.dtype,
            a
        )
        return multiply(a, sb)
    }

    override fun <T : DType, V> divScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val sb = newTensor(
            dataFactory.full<T, V>(a.shape, a.dtype, b),
            a.dtype,
            a
        )
        return divide(a, sb)
    }

    override fun <T : DType, V> rsubScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> {
        val ta = newTensor(
            dataFactory.full<T, V>(b.shape, b.dtype, a),
            b.dtype,
            b
        )
        return subtract(ta, b)
    }

    override fun <T : DType, V> rdivScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> {
        val ta = newTensor(
            dataFactory.full<T, V>(b.shape, b.dtype, a),
            b.dtype,
            b
        )
        return divide(ta, b)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-add")
    override fun <T : DType, V> add(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        return elementwise(a, b) { av, bv, dtype ->
            when (dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = av as Float;
                    val y = bv as Float; (x + y) as V
                }

                sk.ainet.lang.types.Int32::class -> {
                    val x = av as Int;
                    val y = bv as Int; (x + y) as V
                }

                else -> throw IllegalArgumentException("Unsupported dtype for add: ${'$'}dtype")
            }
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-subtract")
    override fun <T : DType, V> subtract(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        return elementwise(a, b) { av, bv, dtype ->
            when (dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = av as Float;
                    val y = bv as Float; (x - y) as V
                }

                sk.ainet.lang.types.Int32::class -> {
                    val x = av as Int;
                    val y = bv as Int; (x - y) as V
                }

                else -> throw IllegalArgumentException("Unsupported dtype for subtract: ${'$'}dtype")
            }
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-multiply")
    override fun <T : DType, V> multiply(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        return elementwise(a, b) { av, bv, dtype ->
            when (dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = av as Float;
                    val y = bv as Float; (x * y) as V
                }

                sk.ainet.lang.types.Int32::class -> {
                    val x = av as Int;
                    val y = bv as Int; (x * y) as V
                }

                else -> throw IllegalArgumentException("Unsupported dtype for multiply: ${'$'}dtype")
            }
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-divide")
    override fun <T : DType, V> divide(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        return elementwise(a, b) { av, bv, dtype ->
            when (dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = av as Float;
                    val y = bv as Float; (x / y) as V
                }

                sk.ainet.lang.types.Int32::class -> {
                    val x = av as Int;
                    val y = bv as Int; if (y == 0) 0 as V else (x / y) as V
                }

                else -> throw IllegalArgumentException("Unsupported dtype for divide: ${'$'}dtype")
            }
        }
    }

    @TensorOp()
    override fun <T : DType, V> matmul(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        require(a.rank >= 1 && b.rank >= 1) { "Matrix multiplication requires tensors with at least 1 dimension per operand" }
        require(a.dtype == b.dtype) { "DType mismatch: ${a.dtype} vs ${b.dtype}" }

        // Fast path: 2D × 2D with FloatArray backing — direct buffer access, no per-element allocation
        if (a.rank == 2 && b.rank == 2
            && (a.dtype == FP32::class)
            && a.data is FloatArrayTensorData<*> && b.data is FloatArrayTensorData<*>
        ) {
            val aBuf = (a.data as FloatArrayTensorData<*>).buffer
            val bBuf = (b.data as FloatArrayTensorData<*>).buffer
            val m = a.shape[0]
            val k = a.shape[1]
            val n = b.shape[1]
            require(k == b.shape[0]) { "Matrix multiplication shape mismatch: ${a.shape} vs ${b.shape}" }
            val out = FloatArray(m * n)
            for (i in 0 until m) {
                val aOff = i * k
                for (j in 0 until n) {
                    var sum = 0f
                    for (p in 0 until k) {
                        sum += aBuf[aOff + p] * bBuf[p * n + j]
                    }
                    out[i * n + j] = sum
                }
            }
            @Suppress("UNCHECKED_CAST")
            val outData = dataFactory.fromFloatArray<T, Float>(Shape(m, n), a.dtype, out) as sk.ainet.lang.tensor.data.TensorData<T, V>
            return newTensor(outData, a.dtype, a, b)
        }

        // Generic fallback for batched / non-float / non-2D cases
        return matmulGeneric(a, b)
    }

    private fun <T : DType, V> matmulGeneric(
        a: Tensor<T, V>,
        b: Tensor<T, V>
    ): Tensor<T, V> {
        val aDims = a.shape.dimensions
        val bDims = b.shape.dimensions
        val aRank = aDims.size
        val bRank = bDims.size
        val aIs1D = aRank == 1
        val bIs1D = bRank == 1

        // Effective shapes (virtually unsqueeze 1D operands):
        val aEff = if (aIs1D) intArrayOf(1, aDims[0]) else aDims
        val bEff = if (bIs1D) intArrayOf(bDims[0], 1) else bDims
        val aEffRank = aEff.size
        val bEffRank = bEff.size

        val kA = aEff[aEffRank - 1]
        val kB = bEff[bEffRank - 2]
        require(kA == kB) { "Matrix multiplication shape mismatch: inner dimensions must match ($kA vs $kB)" }

        // Validate batch dims broadcastability on effective shapes (excluding last two dims)
        val maxEffRank = maxOf(aEffRank, bEffRank)
        for (i in 0 until maxEffRank - 2) {
            val aDim = if (i < aEffRank - 2) aEff[i] else 1
            val bDim = if (i < bEffRank - 2) bEff[i] else 1
            if (aDim != bDim && aDim != 1 && bDim != 1) {
                throw IllegalArgumentException("Matrix multiplication batch dimension mismatch at position $i: $aDim vs $bDim")
            }
        }

        // Compute output shape according to PyTorch rules (squeeze for 1D operands)
        val batchRank = maxEffRank - 2
        val outBatch = IntArray(batchRank) { i ->
            val aDim = if (i < aEffRank - 2) aEff[i] else 1
            val bDim = if (i < bEffRank - 2) bEff[i] else 1
            maxOf(aDim, bDim)
        }
        val m = aEff[aEffRank - 2]
        val n = bEff[bEffRank - 1]

        val outShape = when {
            aIs1D && bIs1D -> Shape(intArrayOf())
            aIs1D -> {
                val dims = IntArray(outBatch.size + 1)
                if (outBatch.isNotEmpty()) outBatch.copyInto(dims, 0)
                dims[dims.size - 1] = n
                Shape(dims)
            }
            bIs1D -> {
                val dims = IntArray(outBatch.size + 1)
                if (outBatch.isNotEmpty()) outBatch.copyInto(dims, 0)
                dims[dims.size - 1] = m
                Shape(dims)
            }
            else -> {
                val dims = IntArray(outBatch.size + 2)
                if (outBatch.isNotEmpty()) outBatch.copyInto(dims, 0)
                dims[dims.size - 2] = m
                dims[dims.size - 1] = n
                Shape(dims)
            }
        }

        fun mapBatchIndexEff(batchIdx: IntArray, effDims: IntArray): IntArray {
            val inBatchRank = effDims.size - 2
            val mapped = IntArray(inBatchRank)
            var or = batchIdx.size - 1
            var ir = inBatchRank - 1
            while (ir >= 0) {
                val inDim = effDims[ir]
                val outIndex = if (or >= 0) batchIdx[or] else 0
                mapped[ir] = if (inDim == 1) 0 else outIndex
                ir--; or--
            }
            return mapped
        }

        val outData = dataFactory.init<T, V>(outShape, a.dtype) { outIdx ->
            val (batchIdx, mIdx, nIdx) = when {
                aIs1D && bIs1D -> Triple(IntArray(0), -1, -1)
                aIs1D -> {
                    val batchLen = outIdx.size - 1
                    val batch = if (batchLen > 0) outIdx.copyOf(batchLen) else IntArray(0)
                    Triple(batch, -1, outIdx.last())
                }
                bIs1D -> {
                    val batchLen = outIdx.size - 1
                    val batch = if (batchLen > 0) outIdx.copyOf(batchLen) else IntArray(0)
                    Triple(batch, outIdx.last(), -1)
                }
                else -> {
                    val batchLen = outIdx.size - 2
                    val batch = if (batchLen > 0) outIdx.copyOf(batchLen) else IntArray(0)
                    Triple(batch, outIdx[outIdx.size - 2], outIdx[outIdx.size - 1])
                }
            }

            val aBatchIdx = mapBatchIndexEff(batchIdx, aEff)
            val bBatchIdx = mapBatchIndexEff(batchIdx, bEff)

            when (a.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    var acc = 0.0f
                    var k = 0
                    while (k < kA) {
                        val av: Float = if (aIs1D) {
                            a.data.get(*intArrayOf(k)) as Float
                        } else {
                            val aIdx = IntArray(aRank)
                            if (aBatchIdx.isNotEmpty()) aBatchIdx.copyInto(aIdx)
                            aIdx[aRank - 2] = mIdx
                            aIdx[aRank - 1] = k
                            a.data.get(*aIdx) as Float
                        }
                        val bv: Float = if (bIs1D) {
                            b.data.get(*intArrayOf(k)) as Float
                        } else {
                            val bIdx = IntArray(bRank)
                            if (bBatchIdx.isNotEmpty()) bBatchIdx.copyInto(bIdx)
                            bIdx[bRank - 2] = k
                            bIdx[bRank - 1] = nIdx
                            b.data.get(*bIdx) as Float
                        }
                        acc += av * bv
                        k++
                    }
                    @Suppress("UNCHECKED_CAST")
                    acc as V
                }

                sk.ainet.lang.types.Int32::class,
                sk.ainet.lang.types.Int8::class -> {
                    var acc = 0
                    var k = 0
                    while (k < kA) {
                        val av: Int = if (aIs1D) {
                            a.data.get(*intArrayOf(k)) as Int
                        } else {
                            val aIdx = IntArray(aRank)
                            if (aBatchIdx.isNotEmpty()) aBatchIdx.copyInto(aIdx)
                            aIdx[aRank - 2] = mIdx
                            aIdx[aRank - 1] = k
                            a.data.get(*aIdx) as Int
                        }
                        val bv: Int = if (bIs1D) {
                            b.data.get(*intArrayOf(k)) as Int
                        } else {
                            val bIdx = IntArray(bRank)
                            if (bBatchIdx.isNotEmpty()) bBatchIdx.copyInto(bIdx)
                            bIdx[bRank - 2] = k
                            bIdx[bRank - 1] = nIdx
                            b.data.get(*bIdx) as Int
                        }
                        acc += av * bv
                        k++
                    }
                    @Suppress("UNCHECKED_CAST")
                    acc as V
                }

                else -> throw IllegalArgumentException("Unsupported dtype for matmul: ${a.dtype}")
            }
        }
        return newTensor(outData, a.dtype, a, b)
    }

    @TensorOp()
    override fun <T : DType, V> transpose(tensor: Tensor<T, V>): Tensor<T, V> {
        val rank = tensor.shape.rank
        require(rank >= 2) { "Transpose requires at least 2 dimensions" }
        val rows = tensor.shape[rank - 2]
        val cols = tensor.shape[rank - 1]

        // Fast path: 2D float tensor — direct buffer swap
        if (rank == 2 && tensor.data is FloatArrayTensorData<*>) {
            val buf = (tensor.data as FloatArrayTensorData<*>).buffer
            val out = FloatArray(rows * cols)
            for (r in 0 until rows) {
                for (c in 0 until cols) {
                    out[c * rows + r] = buf[r * cols + c]
                }
            }
            @Suppress("UNCHECKED_CAST")
            val outData = dataFactory.fromFloatArray<T, Float>(Shape(cols, rows), tensor.dtype, out) as sk.ainet.lang.tensor.data.TensorData<T, V>
            return newTensor(outData, tensor.dtype, tensor)
        }

        // Generic fallback
        val outDims = tensor.shape.dimensions.copyOf()
        outDims[rank - 1] = rows
        outDims[rank - 2] = cols
        val outShape = Shape(outDims)
        val outData = dataFactory.init<T, V>(outShape, tensor.dtype) { outIdx ->
            val inIdx = outIdx.copyOf()
            inIdx[rank - 2] = outIdx[rank - 1]
            inIdx[rank - 1] = outIdx[rank - 2]
            tensor.data.get(*inIdx)
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-conv2d")
    override fun <T : DType, V> conv2d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int
    ): Tensor<T, V> {
        // Validate shapes
        require(input.rank == 4) { "conv2d: input must be 4D (N, C_in, H, W), got ${input.shape.dimensions.contentToString()}" }
        require(weight.rank == 4) { "conv2d: weight must be 4D (C_out, C_in/groups, kH, kW), got ${weight.shape.dimensions.contentToString()}" }
        require(groups >= 1) { "conv2d: groups must be >= 1" }
        require(input.dtype == weight.dtype) { "conv2d: dtype mismatch between input and weight: ${input.dtype} vs ${weight.dtype}" }
        bias?.let { require(it.dtype == input.dtype) { "conv2d: dtype mismatch for bias" } }

        val n = input.shape[0]
        val cIn = input.shape[1]
        val inH = input.shape[2]
        val inW = input.shape[3]

        val cOut = weight.shape[0]
        val cInPerGroup = weight.shape[1]
        val kH = weight.shape[2]
        val kW = weight.shape[3]

        require(cIn % groups == 0) { "conv2d: input channels ${cIn} not divisible by groups ${groups}" }
        require(cOut % groups == 0) { "conv2d: output channels ${cOut} not divisible by groups ${groups}" }
        require(cInPerGroup == cIn / groups) { "conv2d: weight input channels ${cInPerGroup} must equal C_in/groups ${cIn / groups}" }

        val (sH, sW) = stride
        val (pH, pW) = padding
        val (dH, dW) = dilation

        fun outDim(inDim: Int, k: Int, s: Int, p: Int, d: Int): Int {
            return ((inDim + 2 * p - d * (k - 1) - 1) / s) + 1
        }
        val outH = outDim(inH, kH, sH, pH, dH)
        val outW = outDim(inW, kW, sW, pW, dW)
        require(outH >= 0 && outW >= 0) { "conv2d: computed negative output shape (H=${outH}, W=${outW})" }

        // Validate bias shape if provided (accept [C_out] or [1,C_out,1,1])
        if (bias != null) {
            when (bias.rank) {
                1 -> require(bias.shape[0] == cOut) { "conv2d: bias shape must be [C_out], got ${bias.shape.dimensions.contentToString()}" }
                4 -> {
                    require(bias.shape[0] == 1 && bias.shape[1] == cOut && bias.shape[2] == 1 && bias.shape[3] == 1) {
                        "conv2d: bias shape must be [1,C_out,1,1] when 4D, got ${bias.shape.dimensions.contentToString()}"
                    }
                }
                else -> error("conv2d: unsupported bias rank ${bias.rank}")
            }
        }

        val outShape = Shape(n, cOut, outH, outW)
        val outData = dataFactory.init<T, V>(outShape, input.dtype) { outIdx ->
            val bIdx = outIdx[0]
            val oc = outIdx[1]
            val oh = outIdx[2]
            val ow = outIdx[3]

            when (input.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    var acc = 0.0f
                    val groupIdx = (oc * groups) / cOut
                    val inCStart = groupIdx * cInPerGroup
                    val inCEnd = inCStart + cInPerGroup
                    val hBase = oh * sH - pH
                    val wBase = ow * sW - pW

                    var ic = inCStart
                    while (ic < inCEnd) {
                        val kc = ic - inCStart
                        var kh = 0
                        while (kh < kH) {
                            val ih = hBase + kh * dH
                            if (ih >= 0 && ih < inH) {
                                var kw = 0
                                while (kw < kW) {
                                    val iw = wBase + kw * dW
                                    if (iw >= 0 && iw < inW) {
                                        val vIn = input.data.get(bIdx, ic, ih, iw) as Float
                                        val vW = weight.data.get(oc, kc, kh, kw) as Float
                                        acc += vIn * vW
                                    }
                                    kw++
                                }
                            }
                            kh++
                        }
                        ic++
                    }
                    if (bias != null) {
                        val b = when (bias.rank) {
                            1 -> bias.data.get(oc) as Float
                            4 -> bias.data.get(0, oc, 1 - 1, 1 - 1) as Float // [1, C_out, 1, 1]
                            else -> 0.0f
                        }
                        acc += b
                    }
                    @Suppress("UNCHECKED_CAST")
                    acc as V
                }
                sk.ainet.lang.types.Int32::class -> {
                    var acc = 0
                    val groupIdx = (oc * groups) / cOut
                    val inCStart = groupIdx * cInPerGroup
                    val inCEnd = inCStart + cInPerGroup
                    val hBase = oh * sH - pH
                    val wBase = ow * sW - pW

                    var ic = inCStart
                    while (ic < inCEnd) {
                        val kc = ic - inCStart
                        var kh = 0
                        while (kh < kH) {
                            val ih = hBase + kh * dH
                            if (ih >= 0 && ih < inH) {
                                var kw = 0
                                while (kw < kW) {
                                    val iw = wBase + kw * dW
                                    if (iw >= 0 && iw < inW) {
                                        val vIn = input.data.get(bIdx, ic, ih, iw) as Int
                                        val vW = weight.data.get(oc, kc, kh, kw) as Int
                                        acc += vIn * vW
                                    }
                                    kw++
                                }
                            }
                            kh++
                        }
                        ic++
                    }
                    if (bias != null) {
                        val b = when (bias.rank) {
                            1 -> bias.data.get(oc) as Int
                            4 -> bias.data.get(0, oc, 0, 0) as Int
                            else -> 0
                        }
                        acc += b
                    }
                    @Suppress("UNCHECKED_CAST")
                    acc as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for conv2d: ${input.dtype}")
            }
        }
        return if (bias != null) {
            newTensor(outData, input.dtype, input, weight, bias)
        } else {
            newTensor(outData, input.dtype, input, weight)
        }
    }

    @TensorOp()
    override fun <T : DType, V> conv1d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Int,
        padding: Int,
        dilation: Int,
        groups: Int
    ): Tensor<T, V> {
        // Validate shapes
        require(input.rank == 3) { "conv1d: input must be 3D (N, C_in, L), got ${input.shape.dimensions.contentToString()}" }
        require(weight.rank == 3) { "conv1d: weight must be 3D (C_out, C_in/groups, K), got ${weight.shape.dimensions.contentToString()}" }
        require(groups >= 1) { "conv1d: groups must be >= 1" }
        require(input.dtype == weight.dtype) { "conv1d: dtype mismatch between input and weight: ${input.dtype} vs ${weight.dtype}" }
        bias?.let { require(it.dtype == input.dtype) { "conv1d: dtype mismatch for bias" } }

        val n = input.shape[0]
        val cIn = input.shape[1]
        val inL = input.shape[2]

        val cOut = weight.shape[0]
        val cInPerGroup = weight.shape[1]
        val kL = weight.shape[2]

        require(cIn % groups == 0) { "conv1d: input channels ${cIn} not divisible by groups ${groups}" }
        require(cOut % groups == 0) { "conv1d: output channels ${cOut} not divisible by groups ${groups}" }
        require(cInPerGroup == cIn / groups) { "conv1d: weight input channels ${cInPerGroup} must equal C_in/groups ${cIn / groups}" }

        fun outDim(inDim: Int, k: Int, s: Int, p: Int, d: Int): Int {
            return ((inDim + 2 * p - d * (k - 1) - 1) / s) + 1
        }
        val outL = outDim(inL, kL, stride, padding, dilation)
        require(outL >= 0) { "conv1d: computed negative output length (L=${outL})" }

        // Validate bias shape if provided
        if (bias != null) {
            when (bias.rank) {
                1 -> require(bias.shape[0] == cOut) { "conv1d: bias shape must be [C_out], got ${bias.shape.dimensions.contentToString()}" }
                3 -> {
                    require(bias.shape[0] == 1 && bias.shape[1] == cOut && bias.shape[2] == 1) {
                        "conv1d: bias shape must be [1,C_out,1] when 3D, got ${bias.shape.dimensions.contentToString()}"
                    }
                }
                else -> error("conv1d: unsupported bias rank ${bias.rank}")
            }
        }

        val outShape = Shape(n, cOut, outL)
        val outData = dataFactory.init<T, V>(outShape, input.dtype) { outIdx ->
            val bIdx = outIdx[0]
            val oc = outIdx[1]
            val ol = outIdx[2]

            when (input.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    var acc = 0.0f
                    val groupIdx = (oc * groups) / cOut
                    val inCStart = groupIdx * cInPerGroup
                    val inCEnd = inCStart + cInPerGroup
                    val lBase = ol * stride - padding

                    var ic = inCStart
                    while (ic < inCEnd) {
                        val kc = ic - inCStart
                        var kl = 0
                        while (kl < kL) {
                            val il = lBase + kl * dilation
                            if (il >= 0 && il < inL) {
                                val vIn = input.data.get(bIdx, ic, il) as Float
                                val vW = weight.data.get(oc, kc, kl) as Float
                                acc += vIn * vW
                            }
                            kl++
                        }
                        ic++
                    }
                    if (bias != null) {
                        val b = when (bias.rank) {
                            1 -> bias.data.get(oc) as Float
                            3 -> bias.data.get(0, oc, 0) as Float
                            else -> 0.0f
                        }
                        acc += b
                    }
                    @Suppress("UNCHECKED_CAST")
                    acc as V
                }
                sk.ainet.lang.types.Int32::class -> {
                    var acc = 0
                    val groupIdx = (oc * groups) / cOut
                    val inCStart = groupIdx * cInPerGroup
                    val inCEnd = inCStart + cInPerGroup
                    val lBase = ol * stride - padding

                    var ic = inCStart
                    while (ic < inCEnd) {
                        val kc = ic - inCStart
                        var kl = 0
                        while (kl < kL) {
                            val il = lBase + kl * dilation
                            if (il >= 0 && il < inL) {
                                val vIn = input.data.get(bIdx, ic, il) as Int
                                val vW = weight.data.get(oc, kc, kl) as Int
                                acc += vIn * vW
                            }
                            kl++
                        }
                        ic++
                    }
                    if (bias != null) {
                        val b = when (bias.rank) {
                            1 -> bias.data.get(oc) as Int
                            3 -> bias.data.get(0, oc, 0) as Int
                            else -> 0
                        }
                        acc += b
                    }
                    @Suppress("UNCHECKED_CAST")
                    acc as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for conv1d: ${input.dtype}")
            }
        }
        return if (bias != null) {
            newTensor(outData, input.dtype, input, weight, bias)
        } else {
            newTensor(outData, input.dtype, input, weight)
        }
    }

    @TensorOp()
    override fun <T : DType, V> conv3d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Triple<Int, Int, Int>,
        padding: Triple<Int, Int, Int>,
        dilation: Triple<Int, Int, Int>,
        groups: Int
    ): Tensor<T, V> {
        // Validate shapes
        require(input.rank == 5) { "conv3d: input must be 5D (N, C_in, D, H, W), got ${input.shape.dimensions.contentToString()}" }
        require(weight.rank == 5) { "conv3d: weight must be 5D (C_out, C_in/groups, kD, kH, kW), got ${weight.shape.dimensions.contentToString()}" }
        require(groups >= 1) { "conv3d: groups must be >= 1" }
        require(input.dtype == weight.dtype) { "conv3d: dtype mismatch between input and weight: ${input.dtype} vs ${weight.dtype}" }
        bias?.let { require(it.dtype == input.dtype) { "conv3d: dtype mismatch for bias" } }

        val n = input.shape[0]
        val cIn = input.shape[1]
        val inD = input.shape[2]
        val inH = input.shape[3]
        val inW = input.shape[4]

        val cOut = weight.shape[0]
        val cInPerGroup = weight.shape[1]
        val kD = weight.shape[2]
        val kH = weight.shape[3]
        val kW = weight.shape[4]

        require(cIn % groups == 0) { "conv3d: input channels ${cIn} not divisible by groups ${groups}" }
        require(cOut % groups == 0) { "conv3d: output channels ${cOut} not divisible by groups ${groups}" }
        require(cInPerGroup == cIn / groups) { "conv3d: weight input channels ${cInPerGroup} must equal C_in/groups ${cIn / groups}" }

        val (sD, sH, sW) = stride
        val (pD, pH, pW) = padding
        val (dD, dH, dW) = dilation

        fun outDim(inDim: Int, k: Int, s: Int, p: Int, d: Int): Int {
            return ((inDim + 2 * p - d * (k - 1) - 1) / s) + 1
        }
        val outD = outDim(inD, kD, sD, pD, dD)
        val outH = outDim(inH, kH, sH, pH, dH)
        val outW = outDim(inW, kW, sW, pW, dW)
        require(outD >= 0 && outH >= 0 && outW >= 0) { "conv3d: computed negative output shape (D=${outD}, H=${outH}, W=${outW})" }

        // Validate bias shape if provided
        if (bias != null) {
            when (bias.rank) {
                1 -> require(bias.shape[0] == cOut) { "conv3d: bias shape must be [C_out], got ${bias.shape.dimensions.contentToString()}" }
                5 -> {
                    require(bias.shape[0] == 1 && bias.shape[1] == cOut && bias.shape[2] == 1 && bias.shape[3] == 1 && bias.shape[4] == 1) {
                        "conv3d: bias shape must be [1,C_out,1,1,1] when 5D, got ${bias.shape.dimensions.contentToString()}"
                    }
                }
                else -> error("conv3d: unsupported bias rank ${bias.rank}")
            }
        }

        val outShape = Shape(n, cOut, outD, outH, outW)
        val outData = dataFactory.init<T, V>(outShape, input.dtype) { outIdx ->
            val bIdx = outIdx[0]
            val oc = outIdx[1]
            val od = outIdx[2]
            val oh = outIdx[3]
            val ow = outIdx[4]

            when (input.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    var acc = 0.0f
                    val groupIdx = (oc * groups) / cOut
                    val inCStart = groupIdx * cInPerGroup
                    val inCEnd = inCStart + cInPerGroup
                    val dBase = od * sD - pD
                    val hBase = oh * sH - pH
                    val wBase = ow * sW - pW

                    var ic = inCStart
                    while (ic < inCEnd) {
                        val kc = ic - inCStart
                        var kd = 0
                        while (kd < kD) {
                            val id = dBase + kd * dD
                            if (id >= 0 && id < inD) {
                                var kh = 0
                                while (kh < kH) {
                                    val ih = hBase + kh * dH
                                    if (ih >= 0 && ih < inH) {
                                        var kw = 0
                                        while (kw < kW) {
                                            val iw = wBase + kw * dW
                                            if (iw >= 0 && iw < inW) {
                                                val vIn = input.data.get(bIdx, ic, id, ih, iw) as Float
                                                val vW = weight.data.get(oc, kc, kd, kh, kw) as Float
                                                acc += vIn * vW
                                            }
                                            kw++
                                        }
                                    }
                                    kh++
                                }
                            }
                            kd++
                        }
                        ic++
                    }
                    if (bias != null) {
                        val b = when (bias.rank) {
                            1 -> bias.data.get(oc) as Float
                            5 -> bias.data.get(0, oc, 0, 0, 0) as Float
                            else -> 0.0f
                        }
                        acc += b
                    }
                    @Suppress("UNCHECKED_CAST")
                    acc as V
                }
                sk.ainet.lang.types.Int32::class -> {
                    var acc = 0
                    val groupIdx = (oc * groups) / cOut
                    val inCStart = groupIdx * cInPerGroup
                    val inCEnd = inCStart + cInPerGroup
                    val dBase = od * sD - pD
                    val hBase = oh * sH - pH
                    val wBase = ow * sW - pW

                    var ic = inCStart
                    while (ic < inCEnd) {
                        val kc = ic - inCStart
                        var kd = 0
                        while (kd < kD) {
                            val id = dBase + kd * dD
                            if (id >= 0 && id < inD) {
                                var kh = 0
                                while (kh < kH) {
                                    val ih = hBase + kh * dH
                                    if (ih >= 0 && ih < inH) {
                                        var kw = 0
                                        while (kw < kW) {
                                            val iw = wBase + kw * dW
                                            if (iw >= 0 && iw < inW) {
                                                val vIn = input.data.get(bIdx, ic, id, ih, iw) as Int
                                                val vW = weight.data.get(oc, kc, kd, kh, kw) as Int
                                                acc += vIn * vW
                                            }
                                            kw++
                                        }
                                    }
                                    kh++
                                }
                            }
                            kd++
                        }
                        ic++
                    }
                    if (bias != null) {
                        val b = when (bias.rank) {
                            1 -> bias.data.get(oc) as Int
                            5 -> bias.data.get(0, oc, 0, 0, 0) as Int
                            else -> 0
                        }
                        acc += b
                    }
                    @Suppress("UNCHECKED_CAST")
                    acc as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for conv3d: ${input.dtype}")
            }
        }
        return if (bias != null) {
            newTensor(outData, input.dtype, input, weight, bias)
        } else {
            newTensor(outData, input.dtype, input, weight)
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-upsample2d")
    override fun <T : DType, V> upsample2d(
        input: Tensor<T, V>,
        scale: Pair<Int, Int>,
        mode: UpsampleMode,
        alignCorners: Boolean
    ): Tensor<T, V> {
        require(input.rank == 4) { "upsample2d: input must be 4D (N, C, H, W)" }
        val (scaleH, scaleW) = scale
        require(scaleH > 0 && scaleW > 0) { "upsample2d: scale factors must be positive" }
        require(mode == UpsampleMode.Nearest) { "upsample2d: only Nearest mode is implemented on CPU backend" }

        val n = input.shape[0]
        val c = input.shape[1]
        val inH = input.shape[2]
        val inW = input.shape[3]
        val outH = inH * scaleH
        val outW = inW * scaleW
        val outShape = Shape(n, c, outH, outW)

        val outData = dataFactory.init<T, V>(outShape, input.dtype) { idx ->
            val oh = idx[2]
            val ow = idx[3]
            val ih = oh / scaleH
            val iw = ow / scaleW
            input.data.get(idx[0], idx[1], ih, iw)
        }
        return newTensor(outData, input.dtype, input)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-maxpool2d")
    override fun <T : DType, V> maxPool2d(
        input: Tensor<T, V>,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>
    ): Tensor<T, V> {
        require(input.rank == 4) { "maxPool2d: input must be 4D (N, C, H, W)" }
        val n = input.shape[0]
        val c = input.shape[1]
        val inH = input.shape[2]
        val inW = input.shape[3]
        val (kH, kW) = kernelSize
        val (sH, sW) = stride
        val (pH, pW) = padding
        require(kH > 0 && kW > 0) { "maxPool2d: kernel must be > 0" }
        require(sH > 0 && sW > 0) { "maxPool2d: stride must be > 0" }
        fun outDim(inDim: Int, k: Int, s: Int, p: Int): Int = ((inDim + 2 * p - k) / s) + 1
        val outH = outDim(inH, kH, sH, pH)
        val outW = outDim(inW, kW, sW, pW)
        require(outH >= 0 && outW >= 0) { "maxPool2d: negative output size (H=${outH}, W=${outW})" }
        val outShape = Shape(n, c, outH, outW)
        val outData = dataFactory.init<T, V>(outShape, input.dtype) { outIdx ->
            val bIdx = outIdx[0]
            val ch = outIdx[1]
            val oh = outIdx[2]
            val ow = outIdx[3]
            val hBase = oh * sH - pH
            val wBase = ow * sW - pW
            when (input.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    var best = Float.NEGATIVE_INFINITY
                    var kh = 0
                    while (kh < kH) {
                        val ih = hBase + kh
                        if (ih in 0 until inH) {
                            var kw = 0
                            while (kw < kW) {
                                val iw = wBase + kw
                                if (iw in 0 until inW) {
                                    val v = input.data.get(bIdx, ch, ih, iw) as Float
                                    if (v > best) best = v
                                }
                                kw++
                            }
                        }
                        kh++
                    }
                    @Suppress("UNCHECKED_CAST")
                    best as V
                }
                sk.ainet.lang.types.Int32::class, sk.ainet.lang.types.Int8::class -> {
                    var best = Int.MIN_VALUE
                    var kh = 0
                    while (kh < kH) {
                        val ih = hBase + kh
                        if (ih in 0 until inH) {
                            var kw = 0
                            while (kw < kW) {
                                val iw = wBase + kw
                                if (iw in 0 until inW) {
                                    val v = input.data.get(bIdx, ch, ih, iw) as Int
                                    if (v > best) best = v
                                }
                                kw++
                            }
                        }
                        kh++
                    }
                    @Suppress("UNCHECKED_CAST")
                    best as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for maxPool2d: ${input.dtype}")
            }
        }
        return newTensor(outData, input.dtype, input)
    }

    @TensorOp()
    override fun <T : DType, V> avgPool2d(
        input: Tensor<T, V>,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        countIncludePad: Boolean
    ): Tensor<T, V> {
        require(input.rank == 4) { "avgPool2d: input must be 4D (N, C, H, W)" }
        val n = input.shape[0]
        val c = input.shape[1]
        val inH = input.shape[2]
        val inW = input.shape[3]
        val (kH, kW) = kernelSize
        val (sH, sW) = stride
        val (pH, pW) = padding
        require(kH > 0 && kW > 0) { "avgPool2d: kernel must be > 0" }
        require(sH > 0 && sW > 0) { "avgPool2d: stride must be > 0" }
        fun outDim(inDim: Int, k: Int, s: Int, p: Int): Int = ((inDim + 2 * p - k) / s) + 1
        val outH = outDim(inH, kH, sH, pH)
        val outW = outDim(inW, kW, sW, pW)
        require(outH >= 0 && outW >= 0) { "avgPool2d: negative output size (H=${outH}, W=${outW})" }
        val outShape = Shape(n, c, outH, outW)
        val outData = dataFactory.init<T, V>(outShape, input.dtype) { outIdx ->
            val bIdx = outIdx[0]
            val ch = outIdx[1]
            val oh = outIdx[2]
            val ow = outIdx[3]
            val hBase = oh * sH - pH
            val wBase = ow * sW - pW
            when (input.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    var sum = 0.0f
                    var count = 0
                    var kh = 0
                    while (kh < kH) {
                        val ih = hBase + kh
                        var kw = 0
                        while (kw < kW) {
                            val iw = wBase + kw
                            if (ih in 0 until inH && iw in 0 until inW) {
                                val v = input.data.get(bIdx, ch, ih, iw) as Float
                                sum += v
                                count++
                            } else if (countIncludePad) {
                                // Include padding as zero in the count
                                count++
                            }
                            kw++
                        }
                        kh++
                    }
                    // If countIncludePad is true but no valid elements, use kernel size
                    val divisor = if (countIncludePad) (kH * kW) else maxOf(count, 1)
                    @Suppress("UNCHECKED_CAST")
                    (sum / divisor) as V
                }
                sk.ainet.lang.types.Int32::class, sk.ainet.lang.types.Int8::class -> {
                    var sum = 0
                    var count = 0
                    var kh = 0
                    while (kh < kH) {
                        val ih = hBase + kh
                        var kw = 0
                        while (kw < kW) {
                            val iw = wBase + kw
                            if (ih in 0 until inH && iw in 0 until inW) {
                                val v = input.data.get(bIdx, ch, ih, iw) as Int
                                sum += v
                                count++
                            } else if (countIncludePad) {
                                count++
                            }
                            kw++
                        }
                        kh++
                    }
                    val divisor = if (countIncludePad) (kH * kW) else maxOf(count, 1)
                    @Suppress("UNCHECKED_CAST")
                    (sum / divisor) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for avgPool2d: ${input.dtype}")
            }
        }
        return newTensor(outData, input.dtype, input)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-reshape")
    override fun <T : DType, V> reshape(
        tensor: Tensor<T, V>,
        newShape: Shape
    ): Tensor<T, V> {
        // Support -1 dimension inference and validate total elements
        val dims = newShape.dimensions.copyOf()
        var negOneIdx = -1
        var knownProduct = 1
        for (i in dims.indices) {
            val d = dims[i]
            if (d == -1) {
                require(negOneIdx == -1) { "Only one dimension can be -1 in reshape" }
                negOneIdx = i
            } else {
                require(d >= 0) { "Shape dims must be >=0 or -1 for inference: ${d}" }
                knownProduct *= if (d == 0 && dims.size == 0) 1 else d
            }
        }
        val inVol = tensor.shape.volume
        if (negOneIdx >= 0) {
            require(knownProduct != 0) { "Cannot infer dimension with zero known product" }
            require(inVol % knownProduct == 0) { "Cannot infer dimension: input volume ${inVol} not divisible by known product ${knownProduct}" }
            dims[negOneIdx] = inVol / knownProduct
        }
        val finalShape = Shape(dims)
        require(finalShape.volume == inVol) { "Reshape volume mismatch: input=${inVol}, output=${finalShape.volume}" }
        // Reinitialize data by mapping flat index order
        val outData = dataFactory.init<T, V>(finalShape, tensor.dtype) { outIdx ->
            // Compute flat index in output (row-major)
            val outStrides = IntArray(finalShape.rank).apply {
                var s = 1
                for (i in finalShape.rank - 1 downTo 0) {
                    this[i] = s
                    s *= finalShape[i]
                }
            }
            var flat = 0
            for (i in outIdx.indices) flat += outIdx[i] * outStrides[i]
            // Map flat index to input indices using input shape strides
            val inShape = tensor.shape
            val inStrides = IntArray(inShape.rank).apply {
                var s = 1
                for (i in inShape.rank - 1 downTo 0) {
                    this[i] = s
                    s *= inShape[i]
                }
            }
            val inIdx = IntArray(inShape.rank)
            var rem = flat
            for (i in 0 until inShape.rank) {
                inIdx[i] = rem / inStrides[i]
                rem %= inStrides[i]
            }
            tensor.data.get(*inIdx)
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-flatten")
    override fun <T : DType, V> flatten(
        tensor: Tensor<T, V>,
        startDim: Int,
        endDim: Int
    ): Tensor<T, V> {
        val rank = tensor.rank
        require(rank >= 0) { "Invalid tensor rank" }
        fun normDim(d: Int, allowEqRank: Boolean = false): Int {
            val max = if (allowEqRank) rank else rank - 1
            val nd = if (d < 0) d + rank else d
            require(nd in 0..max) { "Dimension out of range: ${d} for rank ${rank}" }
            return nd
        }
        val s = normDim(startDim)
        val e = if (endDim == -1) rank - 1 else normDim(endDim)
        require(s <= e) { "startDim must be <= endDim: start=${s}, end=${e}" }
        if (rank == 0) return tensor // scalar no-op
        // Build new shape
        val newDims = mutableListOf<Int>()
        for (i in 0 until s) newDims += tensor.shape[i]
        var prod = 1
        for (i in s..e) prod *= tensor.shape[i]
        newDims += prod
        for (i in e + 1 until rank) newDims += tensor.shape[i]
        return reshape(tensor, Shape(newDims.toIntArray()))
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-concat")
    override fun <T : DType, V> concat(
        tensors: List<Tensor<T, V>>,
        dim: Int
    ): Tensor<T, V> {
        require(tensors.isNotEmpty()) { "concat: tensors list must not be empty" }
        val first = tensors.first()
        val rank = first.rank
        // Normalize dim allowing dim==rank for scalars to create 1D
        val nd = if (dim < 0) dim + maxOf(rank, 1) else dim
        require(nd >= 0 && nd <= rank) { "concat: dim ${dim} out of range for rank ${rank}" }
        // Allow concatenation along any valid dimension (including dim 0 for rank > 1)
        // Validate shapes and dtype, compute output dims
        var concatSize = 0
        val outDims = IntArray(if (rank == 0) 1 else rank) { i -> if (rank == 0) 0 else first.shape[i] }
        tensors.forEachIndexed { idx, t ->
            require(t.dtype == first.dtype) { "concat: dtype mismatch at tensor ${idx}" }
            if (rank == 0) {
                // scalars: treat as 1D concat
                concatSize += 1
            } else {
                require(t.rank == rank) { "concat: rank mismatch at tensor ${idx}" }
                for (i in 0 until rank) {
                    if (i == nd) continue
                    require(t.shape[i] == first.shape[i]) { "concat: shape mismatch at dim ${i} for tensor ${idx}" }
                }
                concatSize += t.shape[nd]
            }
        }
        if (rank == 0) {
            outDims[0] = concatSize
        } else {
            outDims[nd] = concatSize
        }
        val outShape = Shape(outDims)
        val dtype = first.dtype
        val prefixSums = IntArray(tensors.size + 1)
        for (i in tensors.indices) {
            val sz = if (rank == 0) 1 else tensors[i].shape[nd]
            prefixSums[i + 1] = prefixSums[i] + sz
        }
        val outData = dataFactory.init<T, V>(outShape, dtype) { outIdx ->
            var k = 0
            val along = if (rank == 0) outIdx[0] else outIdx[nd]
            while (k < tensors.size && prefixSums[k + 1] <= along) k++
            val src = tensors[k]
            val localIdx = along - prefixSums[k]
            val inIdx = if (rank == 0) IntArray(0) else outIdx.copyOf()
            if (rank != 0) inIdx[nd] = localIdx
            src.data.get(*inIdx)
        }
        return newTensor(outData, dtype, *tensors.toTypedArray())
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-split")
    override fun <T : DType, V> split(
        tensor: Tensor<T, V>,
        splitSize: Int,
        dim: Int
    ): List<Tensor<T, V>> {
        require(splitSize > 0) { "split: splitSize must be > 0" }
        val rank = tensor.rank
        require(rank >= 0) { "split: invalid rank" }
        val nd = if (dim < 0) dim + rank else dim
        require(nd in 0 until rank) { "split: dim ${dim} out of range for rank ${rank}" }
        val total = tensor.shape[nd]
        val chunks = (total + splitSize - 1) / splitSize
        val result = ArrayList<Tensor<T, V>>(chunks)
        var offset = 0
        for (c in 0 until chunks) {
            val size = minOf(splitSize, total - offset)
            val newDims = tensor.shape.dimensions.copyOf()
            newDims[nd] = size
            val outShape = Shape(newDims)
            val dtype = tensor.dtype
            val outData = dataFactory.init<T, V>(outShape, dtype) { outIdx ->
                val inIdx = outIdx.copyOf()
                inIdx[nd] = inIdx[nd] + offset
                tensor.data.get(*inIdx)
            }
            result += newTensor(outData, dtype, tensor)
            offset += size
        }
        return result
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-squeeze")
    override fun <T : DType, V> squeeze(
        tensor: Tensor<T, V>,
        dim: Int?
    ): Tensor<T, V> {
        val rank = tensor.rank
        require(rank > 0) { "squeeze: tensor must have rank > 0" }
        val dims = tensor.shape.dimensions
        val newDims = if (dim == null) {
            val kept = dims.filter { it != 1 }
            if (kept.isEmpty()) intArrayOf(1) else kept.toIntArray()
        } else {
            val nd = if (dim < 0) dim + rank else dim
            require(nd in 0 until rank) { "squeeze: dim ${dim} out of range for rank ${rank}" }
            require(dims[nd] == 1) { "squeeze: dimension ${dim} must be of size 1" }
            val list = dims.toMutableList()
            list.removeAt(nd)
            if (list.isEmpty()) intArrayOf(1) else list.toIntArray()
        }
        if (newDims.contentEquals(dims)) return tensor
        return reshape(tensor, Shape(newDims))
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-unsqueeze")
    override fun <T : DType, V> unsqueeze(
        tensor: Tensor<T, V>,
        dim: Int
    ): Tensor<T, V> {
        val rank = tensor.rank
        val nd = if (dim < 0) dim + (rank + 1) else dim
        require(nd in 0..rank) { "unsqueeze: dim ${dim} out of range for rank ${rank}" }
        val newDims = IntArray(rank + 1)
        for (i in 0 until nd) newDims[i] = tensor.shape[i]
        newDims[nd] = 1
        for (i in nd until rank) newDims[i + 1] = tensor.shape[i]
        return reshape(tensor, Shape(newDims))
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-relu")
    override fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V> {
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val v = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    (if (v < 0f) 0f else v) as V
                }
                sk.ainet.lang.types.Int32::class, sk.ainet.lang.types.Int8::class -> {
                    val v = tensor.data.get(*idx) as Int
                    @Suppress("UNCHECKED_CAST")
                    (if (v < 0) 0 else v) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for relu: ${'$'}{tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    override fun <T : DType, V> leakyRelu(tensor: Tensor<T, V>, negativeSlope: Float): Tensor<T, V> {
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val v = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    (if (v < 0f) negativeSlope * v else v) as V
                }
                sk.ainet.lang.types.Int32::class, sk.ainet.lang.types.Int8::class -> {
                    val v = tensor.data.get(*idx) as Int
                    @Suppress("UNCHECKED_CAST")
                    (if (v < 0) (negativeSlope * v).toInt() else v) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for leakyRelu: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    override fun <T : DType, V> elu(tensor: Tensor<T, V>, alpha: Float): Tensor<T, V> {
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val v = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    (if (v >= 0f) v else alpha * (kotlin.math.exp(v) - 1f)) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for elu: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-softmax")
    override fun <T : DType, V> softmax(
        tensor: Tensor<T, V>,
        dim: Int
    ): Tensor<T, V> {
        // Stable softmax along dim
        val rank = tensor.rank
        require(rank > 0) { "softmax: tensor must have rank > 0" }
        val nd = if (dim < 0) dim + rank else dim
        require(nd in 0 until rank) { "softmax: dim ${dim} out of range for rank ${rank}" }
        when (tensor.dtype) {
            sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { outIdx ->
                    // For the slice defined by outIdx except for nd, compute softmax at that position
                    val idxMax = outIdx.copyOf()
                    // compute max along nd
                    var maxVal = Float.NEGATIVE_INFINITY
                    for (k in 0 until tensor.shape[nd]) {
                        idxMax[nd] = k
                        val x = tensor.data.get(*idxMax) as Float
                        if (x > maxVal) maxVal = x
                    }
                    // compute denom
                    var denom = 0.0f
                    val idxDen = outIdx.copyOf()
                    for (k in 0 until tensor.shape[nd]) {
                        idxDen[nd] = k
                        val x = tensor.data.get(*idxDen) as Float
                        denom += kotlin.math.exp(x - maxVal)
                    }
                    // numerator for current position
                    val xOut = tensor.data.get(*outIdx) as Float
                    val num = kotlin.math.exp(xOut - maxVal)
                    @Suppress("UNCHECKED_CAST")
                    (num / denom) as V
                }
                return newTensor(outData, tensor.dtype, tensor)
            }
            else -> throw IllegalArgumentException("Unsupported dtype for softmax: ${tensor.dtype}")
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-logsoftmax")
    override fun <T : DType, V> logSoftmax(
        tensor: Tensor<T, V>,
        dim: Int
    ): Tensor<T, V> {
        val rank = tensor.rank
        require(rank > 0) { "logSoftmax: tensor must have rank > 0" }
        val nd = if (dim < 0) dim + rank else dim
        require(nd in 0 until rank) { "logSoftmax: dim ${dim} out of range for rank ${rank}" }
        when (tensor.dtype) {
            sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { outIdx ->
                    val idx = outIdx.copyOf()
                    var maxVal = Float.NEGATIVE_INFINITY
                    for (k in 0 until tensor.shape[nd]) {
                        idx[nd] = k
                        val x = tensor.data.get(*idx) as Float
                        if (x > maxVal) maxVal = x
                    }
                    var sumExp = 0.0f
                    for (k in 0 until tensor.shape[nd]) {
                        idx[nd] = k
                        val x = tensor.data.get(*idx) as Float
                        sumExp += kotlin.math.exp((x - maxVal).toDouble()).toFloat()
                    }
                    val logSumExp = kotlin.math.ln(sumExp.toDouble()).toFloat() + maxVal
                    val xOut = tensor.data.get(*outIdx) as Float
                    @Suppress("UNCHECKED_CAST")
                    (xOut - logSumExp) as V
                }
                return newTensor(outData, tensor.dtype, tensor)
            }
            else -> throw IllegalArgumentException("Unsupported dtype for logSoftmax: ${tensor.dtype}")
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-sigmoid")
    override fun <T : DType, V> sigmoid(tensor: Tensor<T, V>): Tensor<T, V> {
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    (1.0f / (1.0f + kotlin.math.exp(-x))) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for sigmoid: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-silu")
    override fun <T : DType, V> silu(tensor: Tensor<T, V>): Tensor<T, V> {
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = tensor.data.get(*idx) as Float
                    val s = 1.0f / (1.0f + kotlin.math.exp(-x))
                    @Suppress("UNCHECKED_CAST")
                    (x * s) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for silu: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-gelu")
    override fun <T : DType, V> gelu(tensor: Tensor<T, V>): Tensor<T, V> {
        // Tanh approximation of GELU
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val x = tensor.data.get(*idx) as Float
                    val x3 = x * x * x
                    val inner = x + 0.044715f * x3
                    val c = 0.7978845608f // sqrt(2/pi)
                    val t = kotlin.math.tanh(c * inner)
                    val y = 0.5f * x * (1.0f + t)
                    @Suppress("UNCHECKED_CAST")
                    y as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for gelu: ${'$'}{tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-tril")
    override fun <T : DType, V> tril(tensor: Tensor<T, V>, k: Int): Tensor<T, V> {
        val rank = tensor.rank
        // Apply over last two dims; for rank < 2, just copy
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { outIdx ->
            if (rank < 2) {
                @Suppress("UNCHECKED_CAST")
                return@init tensor.data.get(*outIdx) as V
            }
            val rows = tensor.shape[rank - 2]
            val cols = tensor.shape[rank - 1]
            val i = outIdx[rank - 2]
            val j = outIdx[rank - 1]
            val keep = j - i <= k
            if (keep) {
                @Suppress("UNCHECKED_CAST")
                tensor.data.get(*outIdx) as V
            } else {
                when (tensor.dtype) {
                    sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                        @Suppress("UNCHECKED_CAST")
                        0.0f as V
                    }
                    sk.ainet.lang.types.Int32::class, sk.ainet.lang.types.Int8::class -> {
                        @Suppress("UNCHECKED_CAST")
                        0 as V
                    }
                    else -> throw IllegalArgumentException("Unsupported dtype for tril: ${'$'}{tensor.dtype}")
                }
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-sum")
    override fun <T : DType, V> sum(
        tensor: Tensor<T, V>,
        dim: Int?
    ): Tensor<T, V> {
        val rank = tensor.rank
        // Determine reduction mode
        if (dim == null) {
            // Reduce all elements to a scalar (rank-0)
            val outShape = Shape()
            val outData = dataFactory.init<T, V>(outShape, tensor.dtype) {
                when (tensor.dtype) {
                    sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                        var acc = 0.0f
                        val dims = tensor.shape.dimensions
                        if (dims.isEmpty()) {
                            acc = tensor.data.get() as Float
                        } else if (dims.any { it == 0 }) {
                            // Empty tensor: by convention sum over all dims is zero
                            acc = 0.0f
                        } else {
                            val idx = IntArray(dims.size) { 0 }
                            while (true) {
                                acc += tensor.data.get(*idx) as Float
                                var d = dims.size - 1
                                while (d >= 0) {
                                    idx[d]++
                                    if (idx[d] < dims[d]) break
                                    idx[d] = 0
                                    d--
                                }
                                if (d < 0) break
                            }
                        }
                        @Suppress("UNCHECKED_CAST")
                        acc as V
                    }
                    sk.ainet.lang.types.Int32::class -> {
                        var acc = 0
                        val dims = tensor.shape.dimensions
                        if (dims.isEmpty()) {
                            acc = tensor.data.get() as Int
                        } else if (dims.any { it == 0 }) {
                            // Empty tensor: sum is zero
                            acc = 0
                        } else {
                            val idx = IntArray(dims.size) { 0 }
                            while (true) {
                                acc += tensor.data.get(*idx) as Int
                                var d = dims.size - 1
                                while (d >= 0) {
                                    idx[d]++
                                    if (idx[d] < dims[d]) break
                                    idx[d] = 0
                                    d--
                                }
                                if (d < 0) break
                            }
                        }
                        @Suppress("UNCHECKED_CAST")
                        acc as V
                    }
                    else -> throw IllegalArgumentException("Unsupported dtype for sum: ${tensor.dtype}")
                }
            }
            return newTensor(outData, tensor.dtype, tensor)
        } else {
            val nd = if (dim < 0) dim + rank else dim
            require(nd in 0 until rank) { "sum: dim ${dim} out of range for rank ${rank}" }
            val inDims = tensor.shape.dimensions
            val outDims = IntArray(rank - 1) { 0 }
            var oi = 0
            for (i in 0 until rank) {
                if (i == nd) continue
                outDims[oi++] = inDims[i]
            }
            val outShape = Shape(outDims)
            val outData = dataFactory.init<T, V>(outShape, tensor.dtype) { outIdx ->
                when (tensor.dtype) {
                    sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                        var acc = 0.0f
                        val inIdx = IntArray(rank)
                        var o = 0
                        for (i in 0 until rank) {
                            if (i == nd) continue
                            inIdx[i] = outIdx[o++]
                        }
                        for (k in 0 until inDims[nd]) {
                            inIdx[nd] = k
                            acc += tensor.data.get(*inIdx) as Float
                        }
                        @Suppress("UNCHECKED_CAST")
                        acc as V
                    }
                    sk.ainet.lang.types.Int32::class -> {
                        var acc = 0
                        val inIdx = IntArray(rank)
                        var o = 0
                        for (i in 0 until rank) {
                            if (i == nd) continue
                            inIdx[i] = outIdx[o++]
                        }
                        for (k in 0 until inDims[nd]) {
                            inIdx[nd] = k
                            acc += tensor.data.get(*inIdx) as Int
                        }
                        @Suppress("UNCHECKED_CAST")
                        acc as V
                    }
                    else -> throw IllegalArgumentException("Unsupported dtype for sum: ${tensor.dtype}")
                }
            }
            return newTensor(outData, tensor.dtype, tensor)
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-mean")
    override fun <T : DType, V> mean(
        tensor: Tensor<T, V>,
        dim: Int?
    ): Tensor<T, V> {
        val rank = tensor.rank
        if (dim == null) {
            val outShape = Shape()
            val count = if (tensor.volume == 0) 0 else tensor.volume
            val outData = dataFactory.init<T, V>(outShape, tensor.dtype) {
                when (tensor.dtype) {
                    sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                        if (count == 0) {
                            @Suppress("UNCHECKED_CAST") (0.0f as V)
                        } else {
                            var acc = 0.0f
                            val dims = tensor.shape.dimensions
                            if (dims.isEmpty()) {
                                acc = tensor.data.get() as Float
                            } else {
                                val idx = IntArray(dims.size)
                                while (true) {
                                    acc += tensor.data.get(*idx) as Float
                                    var d = dims.size - 1
                                    while (d >= 0) {
                                        idx[d]++
                                        if (idx[d] < dims[d]) break
                                        idx[d] = 0
                                        d--
                                    }
                                    if (d < 0) break
                                }
                            }
                            @Suppress("UNCHECKED_CAST") (acc / count.toFloat()) as V
                        }
                    }
                    sk.ainet.lang.types.Int32::class -> {
                        if (count == 0) {
                            @Suppress("UNCHECKED_CAST") (0 as V)
                        } else {
                            var acc = 0
                            val dims = tensor.shape.dimensions
                            if (dims.isEmpty()) {
                                acc = tensor.data.get() as Int
                            } else {
                                val idx = IntArray(dims.size)
                                while (true) {
                                    acc += tensor.data.get(*idx) as Int
                                    var d = dims.size - 1
                                    while (d >= 0) {
                                        idx[d]++
                                        if (idx[d] < dims[d]) break
                                        idx[d] = 0
                                        d--
                                    }
                                    if (d < 0) break
                                }
                            }
                            @Suppress("UNCHECKED_CAST") (acc / count) as V
                        }
                    }
                    else -> throw IllegalArgumentException("Unsupported dtype for mean: ${tensor.dtype}")
                }
            }
            return newTensor(outData, tensor.dtype, tensor)
        } else {
            val nd = if (dim < 0) dim + rank else dim
            require(nd in 0 until rank) { "mean: dim ${dim} out of range for rank ${rank}" }
            val inDims = tensor.shape.dimensions
            val outDims = IntArray(rank - 1)
            var oi = 0
            for (i in 0 until rank) {
                if (i == nd) continue
                outDims[oi++] = inDims[i]
            }
            val outShape = Shape(outDims)
            val n = inDims[nd]
            val outData = dataFactory.init<T, V>(outShape, tensor.dtype) { outIdx ->
                when (tensor.dtype) {
                    sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                        if (n == 0) {
                            @Suppress("UNCHECKED_CAST") (0.0f as V)
                        } else {
                            var acc = 0.0f
                            val inIdx = IntArray(rank)
                            var o = 0
                            for (i in 0 until rank) {
                                if (i == nd) continue
                                inIdx[i] = outIdx[o++]
                            }
                            for (k in 0 until n) {
                                inIdx[nd] = k
                                acc += tensor.data.get(*inIdx) as Float
                            }
                            @Suppress("UNCHECKED_CAST") (acc / n.toFloat()) as V
                        }
                    }
                    sk.ainet.lang.types.Int32::class -> {
                        if (n == 0) {
                            @Suppress("UNCHECKED_CAST") (0 as V)
                        } else {
                            var acc = 0
                            val inIdx = IntArray(rank)
                            var o = 0
                            for (i in 0 until rank) {
                                if (i == nd) continue
                                inIdx[i] = outIdx[o++]
                            }
                            for (k in 0 until n) {
                                inIdx[nd] = k
                                acc += tensor.data.get(*inIdx) as Int
                            }
                            @Suppress("UNCHECKED_CAST") (acc / n) as V
                        }
                    }
                    else -> throw IllegalArgumentException("Unsupported dtype for mean: ${tensor.dtype}")
                }
            }
            return newTensor(outData, tensor.dtype, tensor)
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-variance")
    override fun <T : DType, V> variance(
        tensor: Tensor<T, V>,
        dim: Int?
    ): Tensor<T, V> {
        val rank = tensor.rank
        if (dim == null) {
            val outShape = Shape()
            val n = tensor.volume
            val outData = dataFactory.init<T, V>(outShape, tensor.dtype) {
                when (tensor.dtype) {
                    sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                        if (n == 0) {
                            @Suppress("UNCHECKED_CAST") (0.0f as V)
                        } else {
                            var sum = 0.0f
                            var sumSq = 0.0f
                            val dims = tensor.shape.dimensions
                            if (dims.isEmpty()) {
                                val x = tensor.data.get() as Float
                                sum = x
                                sumSq = x * x
                            } else {
                                val idx = IntArray(dims.size)
                                while (true) {
                                    val x = tensor.data.get(*idx) as Float
                                    sum += x
                                    sumSq += x * x
                                    var d = dims.size - 1
                                    while (d >= 0) {
                                        idx[d]++
                                        if (idx[d] < dims[d]) break
                                        idx[d] = 0
                                        d--
                                    }
                                    if (d < 0) break
                                }
                            }
                            val mean = sum / n.toFloat()
                            val varVal = (sumSq / n.toFloat()) - mean * mean
                            @Suppress("UNCHECKED_CAST") varVal as V
                        }
                    }
                    sk.ainet.lang.types.Int32::class -> {
                        if (n == 0) {
                            @Suppress("UNCHECKED_CAST") (0 as V)
                        } else {
                            var sum = 0L
                            var sumSq = 0L
                            val dims = tensor.shape.dimensions
                            if (dims.isEmpty()) {
                                val x = (tensor.data.get() as Int).toLong()
                                sum = x
                                sumSq = x * x
                            } else {
                                val idx = IntArray(dims.size)
                                while (true) {
                                    val x = (tensor.data.get(*idx) as Int).toLong()
                                    sum += x
                                    sumSq += x * x
                                    var d = dims.size - 1
                                    while (d >= 0) {
                                        idx[d]++
                                        if (idx[d] < dims[d]) break
                                        idx[d] = 0
                                        d--
                                    }
                                    if (d < 0) break
                                }
                            }
                            val meanNum = sum / n.toLong()
                            val varNum = (sumSq / n.toLong()) - meanNum * meanNum
                            @Suppress("UNCHECKED_CAST") (varNum.toInt() as V)
                        }
                    }
                    else -> throw IllegalArgumentException("Unsupported dtype for variance: ${tensor.dtype}")
                }
            }
            return newTensor(outData, tensor.dtype, tensor)
        } else {
            val nd = if (dim < 0) dim + rank else dim
            require(nd in 0 until rank) { "variance: dim ${dim} out of range for rank ${rank}" }
            val inDims = tensor.shape.dimensions
            val outDims = IntArray(rank - 1)
            var oi = 0
            for (i in 0 until rank) {
                if (i == nd) continue
                outDims[oi++] = inDims[i]
            }
            val outShape = Shape(outDims)
            val n = inDims[nd]
            val outData = dataFactory.init<T, V>(outShape, tensor.dtype) { outIdx ->
                when (tensor.dtype) {
                    sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                        if (n == 0) {
                            @Suppress("UNCHECKED_CAST") (0.0f as V)
                        } else {
                            var sum = 0.0f
                            var sumSq = 0.0f
                            val inIdx = IntArray(rank)
                            var o = 0
                            for (i in 0 until rank) {
                                if (i == nd) continue
                                inIdx[i] = outIdx[o++]
                            }
                            for (k in 0 until n) {
                                inIdx[nd] = k
                                val x = tensor.data.get(*inIdx) as Float
                                sum += x
                                sumSq += x * x
                            }
                            val mean = sum / n.toFloat()
                            val varVal = (sumSq / n.toFloat()) - mean * mean
                            @Suppress("UNCHECKED_CAST") varVal as V
                        }
                    }
                    sk.ainet.lang.types.Int32::class -> {
                        if (n == 0) {
                            @Suppress("UNCHECKED_CAST") (0 as V)
                        } else {
                            var sum = 0L
                            var sumSq = 0L
                            val inIdx = IntArray(rank)
                            var o = 0
                            for (i in 0 until rank) {
                                if (i == nd) continue
                                inIdx[i] = outIdx[o++]
                            }
                            for (k in 0 until n) {
                                inIdx[nd] = k
                                val x = (tensor.data.get(*inIdx) as Int).toLong()
                                sum += x
                                sumSq += x * x
                            }
                            val meanNum = sum / n.toLong()
                            val varNum = (sumSq / n.toLong()) - meanNum * meanNum
                            @Suppress("UNCHECKED_CAST") (varNum.toInt() as V)
                        }
                    }
                    else -> throw IllegalArgumentException("Unsupported dtype for variance: ${tensor.dtype}")
                }
            }
            return newTensor(outData, tensor.dtype, tensor)
        }
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-sqrt")
    override fun <T : DType, V> sqrt(tensor: Tensor<T, V>): Tensor<T, V> {
        require(
            tensor.dtype == sk.ainet.lang.types.FP32::class ||
                tensor.dtype == sk.ainet.lang.types.FP16::class
        ) { "sqrt supports only FP16/FP32, got ${tensor.dtype}" }

        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            val v = tensor.data.get(*idx) as Float
            @Suppress("UNCHECKED_CAST")
            sqrt(v) as V
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    // ---- TinyFoA ops: abs, sign, clamp, lt, ge ----

    @TensorOp()
    @InProgress("cpu", owner = "team:tinyfoa", issue = "PRD-tinyFoA#op-abs")
    override fun <T : DType, V> abs(tensor: Tensor<T, V>): Tensor<T, V> {
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val v = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    kotlin.math.abs(v) as V
                }
                sk.ainet.lang.types.Int32::class -> {
                    val v = tensor.data.get(*idx) as Int
                    @Suppress("UNCHECKED_CAST")
                    kotlin.math.abs(v) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for abs: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:tinyfoa", issue = "PRD-tinyFoA#op-sign")
    override fun <T : DType, V> sign(tensor: Tensor<T, V>): Tensor<T, V> {
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val v = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    (if (v > 0f) 1f else if (v < 0f) -1f else 0f) as V
                }
                sk.ainet.lang.types.Int32::class -> {
                    val v = tensor.data.get(*idx) as Int
                    @Suppress("UNCHECKED_CAST")
                    (if (v > 0) 1 else if (v < 0) -1 else 0) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for sign: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:tinyfoa", issue = "PRD-tinyFoA#op-clamp")
    override fun <T : DType, V> clamp(tensor: Tensor<T, V>, minVal: Float, maxVal: Float): Tensor<T, V> {
        require(minVal <= maxVal) { "clamp: minVal ($minVal) must be <= maxVal ($maxVal)" }
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val v = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    v.coerceIn(minVal, maxVal) as V
                }
                sk.ainet.lang.types.Int32::class -> {
                    val v = tensor.data.get(*idx) as Int
                    @Suppress("UNCHECKED_CAST")
                    v.coerceIn(minVal.toInt(), maxVal.toInt()) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for clamp: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:tinyfoa", issue = "PRD-tinyFoA#op-lt")
    override fun <T : DType, V> lt(tensor: Tensor<T, V>, value: Float): Tensor<T, V> {
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val v = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    (if (v < value) 1f else 0f) as V
                }
                sk.ainet.lang.types.Int32::class -> {
                    val v = tensor.data.get(*idx) as Int
                    @Suppress("UNCHECKED_CAST")
                    (if (v < value.toInt()) 1 else 0) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for lt: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:tinyfoa", issue = "PRD-tinyFoA#op-ge")
    override fun <T : DType, V> ge(tensor: Tensor<T, V>, value: Float): Tensor<T, V> {
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            when (tensor.dtype) {
                sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                    val v = tensor.data.get(*idx) as Float
                    @Suppress("UNCHECKED_CAST")
                    (if (v >= value) 1f else 0f) as V
                }
                sk.ainet.lang.types.Int32::class -> {
                    val v = tensor.data.get(*idx) as Int
                    @Suppress("UNCHECKED_CAST")
                    (if (v >= value.toInt()) 1 else 0) as V
                }
                else -> throw IllegalArgumentException("Unsupported dtype for ge: ${tensor.dtype}")
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    // ---- narrow, pad2d, unfold ----

    @TensorOp()
    @InProgress("cpu", owner = "team:tinyfoa", issue = "PRD-tinyFoA#op-narrow")
    override fun <T : DType, V> narrow(tensor: Tensor<T, V>, dim: Int, start: Int, length: Int): Tensor<T, V> {
        val actualDim = if (dim < 0) tensor.shape.rank + dim else dim
        require(actualDim in 0 until tensor.shape.rank) { "narrow dim $dim out of bounds for rank ${tensor.shape.rank}" }
        require(start >= 0 && start + length <= tensor.shape.dimensions[actualDim]) {
            "narrow: start=$start length=$length exceeds dim size ${tensor.shape.dimensions[actualDim]}"
        }
        val resultDims = tensor.shape.dimensions.copyOf()
        resultDims[actualDim] = length
        val outShape = Shape(resultDims)
        val outData = dataFactory.init<T, V>(outShape, tensor.dtype) { idx ->
            val srcIdx = idx.copyOf()
            srcIdx[actualDim] = srcIdx[actualDim] + start
            tensor.data.get(*srcIdx)
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:tinyfoa", issue = "PRD-tinyFoA#op-pad2d")
    override fun <T : DType, V> pad2d(tensor: Tensor<T, V>, padLeft: Int, padRight: Int, padTop: Int, padBottom: Int): Tensor<T, V> {
        require(tensor.shape.rank == 4) { "pad2d requires 4D tensor [N,C,H,W], got rank ${tensor.shape.rank}" }
        val (n, c, h, w) = tensor.shape.dimensions.toList()
        val newH = h + padTop + padBottom
        val newW = w + padLeft + padRight
        val outShape = Shape(n, c, newH, newW)
        val outData = dataFactory.init<T, V>(outShape, tensor.dtype) { idx ->
            val srcRow = idx[2] - padTop
            val srcCol = idx[3] - padLeft
            if (srcRow in 0 until h && srcCol in 0 until w) {
                tensor.data.get(idx[0], idx[1], srcRow, srcCol)
            } else {
                // Zero padding
                when (tensor.dtype) {
                    sk.ainet.lang.types.FP32::class, sk.ainet.lang.types.FP16::class -> {
                        @Suppress("UNCHECKED_CAST") (0f as V)
                    }
                    sk.ainet.lang.types.Int32::class -> {
                        @Suppress("UNCHECKED_CAST") (0 as V)
                    }
                    else -> throw IllegalArgumentException("Unsupported dtype for pad2d: ${tensor.dtype}")
                }
            }
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:tinyfoa", issue = "PRD-tinyFoA#op-unfold")
    override fun <T : DType, V> unfold(tensor: Tensor<T, V>, dim: Int, size: Int, step: Int): Tensor<T, V> {
        val actualDim = if (dim < 0) tensor.shape.rank + dim else dim
        require(actualDim in 0 until tensor.shape.rank) { "unfold dim $dim out of bounds for rank ${tensor.shape.rank}" }
        val dimSize = tensor.shape.dimensions[actualDim]
        require(size <= dimSize) { "unfold size $size > dim size $dimSize" }
        val numWindows = (dimSize - size) / step + 1
        val resultDims = IntArray(tensor.shape.rank + 1)
        for (i in 0 until tensor.shape.rank) {
            resultDims[i] = if (i == actualDim) numWindows else tensor.shape.dimensions[i]
        }
        resultDims[tensor.shape.rank] = size
        val outShape = Shape(resultDims)
        val outData = dataFactory.init<T, V>(outShape, tensor.dtype) { idx ->
            // idx has rank+1 dimensions. Last dimension is the window element index.
            val windowIdx = idx[tensor.shape.rank]
            val srcIdx = IntArray(tensor.shape.rank)
            for (i in 0 until tensor.shape.rank) {
                srcIdx[i] = if (i == actualDim) {
                    idx[i] * step + windowIdx
                } else {
                    idx[i]
                }
            }
            tensor.data.get(*srcIdx)
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    @TensorOp()
    @InProgress("cpu", owner = "team:cpu", issue = "task-ops.md#op-convert")
    override fun <TFrom : DType, TTo : DType, V> convert(
        tensor: Tensor<TFrom, V>,
        targetType: TTo
    ): Tensor<TTo, V> {
        TODO("Not yet implemented")
    }

    override fun <T : DType, V> gather(input: Tensor<T, V>, indices: Tensor<DType, *>, dim: Int): Tensor<T, V> {
        require(dim == 0) { "gather: only dim=0 supported currently, got dim=$dim" }
        // Gather rows from input along dimension 0 using indices
        // Input: [vocabSize, embeddingDim], Indices: [L] or [N, L]
        // Output: [L, embeddingDim] or [N, L, embeddingDim]
        val numIndices = indices.volume
        val indexList = IntArray(numIndices) { i ->
            val v = indices.data[i]
            (v as Number).toInt()
        }

        if (input.rank == 2) {
            val embDim = input.shape[1]
            val outShape = if (indices.rank == 1) {
                Shape(intArrayOf(numIndices, embDim))
            } else {
                // Preserve index shape + embedding dim
                Shape(IntArray(indices.rank) { indices.shape[it] } + intArrayOf(embDim))
            }
            val outData = dataFactory.init<T, V>(outShape, input.dtype) { outIdx ->
                // Map multi-dim output index to flat index and embedding position
                val flatIdx = if (outIdx.size == 2) outIdx[0] else {
                    var flat = 0
                    for (d in 0 until outIdx.size - 1) {
                        flat = flat * (if (d < indices.rank) indices.shape[d] else 1) + outIdx[d]
                    }
                    flat
                }
                val row = indexList[flatIdx]
                val col = outIdx[outIdx.size - 1]
                input.data[row, col]
            }
            return newTensor(outData, input.dtype, input)
        }

        // Fallback for higher-rank inputs
        error("gather: unsupported input rank ${input.rank}")
    }

    override fun <T : DType, V> indexSelect(input: Tensor<T, V>, indices: Tensor<DType, *>, dim: Int): Tensor<T, V> {
        require(dim in 0 until input.rank) { "indexSelect: dim=$dim out of range for rank ${input.rank}" }
        val numIndices = indices.volume
        val indexList = IntArray(numIndices) { i ->
            val v = indices.data[i]
            (v as Number).toInt()
        }

        val resultDims = input.shape.dimensions.copyOf()
        resultDims[dim] = numIndices
        val resultShape = Shape(resultDims)

        val outData = dataFactory.init<T, V>(resultShape, input.dtype) { outIdx ->
            // Replace the dim-th index with the looked-up value from indexList
            val srcIdx = outIdx.copyOf()
            srcIdx[dim] = indexList[outIdx[dim]]
            input.data.get(*srcIdx)
        }
        return newTensor(outData, input.dtype, input)
    }

    override fun <T : DType, V> exp(tensor: Tensor<T, V>): Tensor<T, V> {
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            val x = tensor.data.get(*idx) as Float
            @Suppress("UNCHECKED_CAST")
            kotlin.math.exp(x) as V
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    override fun <T : DType, V> expm1(tensor: Tensor<T, V>): Tensor<T, V> {
        val outData = dataFactory.init<T, V>(tensor.shape, tensor.dtype) { idx ->
            val x = tensor.data.get(*idx) as Float
            @Suppress("UNCHECKED_CAST")
            kotlin.math.expm1(x.toDouble()).toFloat() as V
        }
        return newTensor(outData, tensor.dtype, tensor)
    }

    override fun <T : DType, V> scaledDotProductAttention(
        query: Tensor<T, V>,
        key: Tensor<T, V>,
        value: Tensor<T, V>,
        mask: Tensor<T, V>?,
        scale: Float,
        causal: Boolean
    ): Tensor<T, V> {
        // Expected shapes: [batch, heads, seqQ, headDim] for Q, [batch, heads, seqKV, headDim] for K/V
        require(query.rank == 4) { "SDPA: expected rank-4 query, got rank ${query.rank}" }
        require(key.rank == 4) { "SDPA: expected rank-4 key, got rank ${key.rank}" }
        require(value.rank == 4) { "SDPA: expected rank-4 value, got rank ${value.rank}" }

        val batch = query.shape[0]
        val heads = query.shape[1]
        val seqQ = query.shape[2]
        val headDim = query.shape[3]
        val seqKV = key.shape[2]

        val qBuf = query.data.copyToFloatArray()
        val kBuf = key.data.copyToFloatArray()
        val vBuf = value.data.copyToFloatArray()

        val outBuf = FloatArray(batch * heads * seqQ * headDim)

        for (b in 0 until batch) {
            for (h in 0 until heads) {
                // Compute attention scores: Q @ K^T, then scale
                val scores = FloatArray(seqQ * seqKV)
                for (qi in 0 until seqQ) {
                    for (ki in 0 until seqKV) {
                        var dot = 0f
                        val qOff = ((b * heads + h) * seqQ + qi) * headDim
                        val kOff = ((b * heads + h) * seqKV + ki) * headDim
                        for (d in 0 until headDim) {
                            dot += qBuf[qOff + d] * kBuf[kOff + d]
                        }
                        scores[qi * seqKV + ki] = dot * scale
                    }
                }

                // Apply causal mask (set future positions to -inf)
                if (causal) {
                    for (qi in 0 until seqQ) {
                        // For single-token inference with KV cache: qi indexes from the
                        // query's perspective. The valid range of ki is [0, seqKV).
                        // With cache, seqQ=1 and seqKV=pos+1, so all keys are valid (no masking needed).
                        // Without cache, seqQ=seqKV, and we mask ki > qi.
                        val maxKi = if (seqQ == seqKV) qi else seqKV - seqQ + qi
                        for (ki in maxKi + 1 until seqKV) {
                            scores[qi * seqKV + ki] = Float.NEGATIVE_INFINITY
                        }
                    }
                }

                // Apply external mask if provided
                if (mask != null) {
                    val maskBuf = mask.data.copyToFloatArray()
                    for (i in scores.indices) {
                        scores[i] += maskBuf[i % maskBuf.size]
                    }
                }

                // Softmax over the key dimension for each query position
                for (qi in 0 until seqQ) {
                    val off = qi * seqKV
                    var maxVal = Float.NEGATIVE_INFINITY
                    for (ki in 0 until seqKV) {
                        if (scores[off + ki] > maxVal) maxVal = scores[off + ki]
                    }
                    var sumExp = 0f
                    for (ki in 0 until seqKV) {
                        val e = kotlin.math.exp(scores[off + ki] - maxVal)
                        scores[off + ki] = e
                        sumExp += e
                    }
                    if (sumExp > 0f) {
                        for (ki in 0 until seqKV) {
                            scores[off + ki] /= sumExp
                        }
                    }
                }

                // Compute output: scores @ V
                for (qi in 0 until seqQ) {
                    val outOff = ((b * heads + h) * seqQ + qi) * headDim
                    for (d in 0 until headDim) {
                        var sum = 0f
                        for (ki in 0 until seqKV) {
                            val vOff = ((b * heads + h) * seqKV + ki) * headDim
                            sum += scores[qi * seqKV + ki] * vBuf[vOff + d]
                        }
                        outBuf[outOff + d] = sum
                    }
                }
            }
        }

        val shape = Shape(batch, heads, seqQ, headDim)
        @Suppress("UNCHECKED_CAST")
        val data = dataFactory.fromFloatArray<T, Float>(shape, query.dtype, outBuf) as sk.ainet.lang.tensor.data.TensorData<T, V>
        return newTensor(data, query.dtype, query, key, value)
    }

}

public class DefaultCpuOps(dataFactory: TensorDataFactory) : DefaultCpuOpsBase(dataFactory)
