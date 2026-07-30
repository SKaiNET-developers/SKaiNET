package sk.ainet.lang.tensor.ops

import sk.ainet.lang.ops.Backend
import sk.ainet.lang.ops.InProgress
import sk.ainet.lang.tensor.Dim
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.hasDynamic
import sk.ainet.lang.tensor.data.DenseTensorDataFactory
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import sk.ainet.lang.tensor.data.views.UnsqueezedTensorData
import kotlin.reflect.KClass

@Backend(id = "void", displayName = "Shape-only", internal = true)
public class VoidTensorOps : TensorOps {

    // Shape-only tracing. A STATIC shape still gets a real (readable) zeros buffer — existing code
    // creates void tensors and reads their zeros, so that behavior must be preserved. A DYNAMIC shape
    // (a `Dim.DYNAMIC` extent) cannot be allocated at all (it would throw NegativeArraySizeException),
    // so it gets an allocation-free ShapeOnlyTensorData that carries only the Shape — which is exactly
    // what lets a dynamic KV-cache seq dim thread through a decode trace. See ShapeOnlyDataFactory.
    private val dataFactory = ShapeOnlyDataFactory
    
    /**
     * Validates that two shapes are compatible for element-wise operations.
     * Implements NumPy-style broadcasting rules.
     */
    private fun validateElementWiseShapes(a: Shape, b: Shape, operation: String) {
        if (!areShapesBroadcastable(a, b)) {
            throw IllegalArgumentException(
                "Shape mismatch for $operation: ${a.dimensions.contentToString()} vs ${b.dimensions.contentToString()}"
            )
        }
    }
    
    /**
     * Checks if two shapes are broadcastable according to NumPy broadcasting rules.
     * Two shapes are broadcastable if:
     * 1. They have the same number of dimensions, or
     * 2. One can be broadcast to match the other by prepending 1s to the smaller shape
     * 3. For each dimension, the sizes are equal or one of them is 1
     */
    private fun areShapesBroadcastable(a: Shape, b: Shape): Boolean {
        val aDims = a.dimensions
        val bDims = b.dimensions
        val maxLen = maxOf(aDims.size, bDims.size)
        
        // Pad shorter shape with 1s at the beginning
        val aPadded = IntArray(maxLen) { i -> 
            if (i < maxLen - aDims.size) 1 else aDims[i - (maxLen - aDims.size)] 
        }
        val bPadded = IntArray(maxLen) { i -> 
            if (i < maxLen - bDims.size) 1 else bDims[i - (maxLen - bDims.size)] 
        }
        
        // Check broadcasting compatibility for each dimension
        for (i in 0 until maxLen) {
            if (aPadded[i] != bPadded[i] && aPadded[i] != 1 && bPadded[i] != 1) {
                return false
            }
        }
        return true
    }
    
    /**
     * Calculates the result shape after broadcasting two shapes.
     * The result shape has the maximum size for each dimension.
     */
    private fun calculateBroadcastShape(a: Shape, b: Shape): Shape {
        val aDims = a.dimensions
        val bDims = b.dimensions
        val maxLen = maxOf(aDims.size, bDims.size)
        
        // Pad shorter shape with 1s at the beginning
        val aPadded = IntArray(maxLen) { i -> 
            if (i < maxLen - aDims.size) 1 else aDims[i - (maxLen - aDims.size)] 
        }
        val bPadded = IntArray(maxLen) { i -> 
            if (i < maxLen - bDims.size) 1 else bDims[i - (maxLen - bDims.size)] 
        }
        
        // Result shape takes the maximum of each dimension
        val resultDims = IntArray(maxLen) { i -> maxOf(aPadded[i], bPadded[i]) }
        return Shape(resultDims)
    }
    
    override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        validateElementWiseShapes(a.shape, b.shape, "addition")
        val resultShape = calculateBroadcastShape(a.shape, b.shape)
        val resultData = dataFactory.zeros<T, V>(resultShape, a.dtype)
        return VoidOpsTensor(resultData, a.dtype)
    }

    override fun <T : DType, V> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        validateElementWiseShapes(a.shape, b.shape, "subtraction")
        val resultShape = calculateBroadcastShape(a.shape, b.shape)
        val resultData = dataFactory.zeros<T, V>(resultShape, a.dtype)
        return VoidOpsTensor(resultData, a.dtype)
    }

    override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        validateElementWiseShapes(a.shape, b.shape, "multiplication")
        val resultShape = calculateBroadcastShape(a.shape, b.shape)
        val resultData = dataFactory.zeros<T, V>(resultShape, a.dtype)
        return VoidOpsTensor(resultData, a.dtype)
    }

    override fun <T : DType, V> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        validateElementWiseShapes(a.shape, b.shape, "division")
        val resultShape = calculateBroadcastShape(a.shape, b.shape)
        val resultData = dataFactory.zeros<T, V>(resultShape, a.dtype)
        return VoidOpsTensor(resultData, a.dtype)
    }

    // Scalar operations: return zero tensors with the same shape as the tensor operand
    override fun <T : DType, V> addScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(a.shape, a.dtype)
        return VoidOpsTensor(resultData, a.dtype)
    }

    override fun <T : DType, V> subScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(a.shape, a.dtype)
        return VoidOpsTensor(resultData, a.dtype)
    }

    override fun <T : DType, V> mulScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(a.shape, a.dtype)
        return VoidOpsTensor(resultData, a.dtype)
    }

    override fun <T : DType, V> divScalar(a: Tensor<T, V>, b: Number): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(a.shape, a.dtype)
        return VoidOpsTensor(resultData, a.dtype)
    }

    override fun <T : DType, V> rsubScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(b.shape, b.dtype)
        return VoidOpsTensor(resultData, b.dtype)
    }

    override fun <T : DType, V> rdivScalar(a: Number, b: Tensor<T, V>): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(b.shape, b.dtype)
        return VoidOpsTensor(resultData, b.dtype)
    }

    @InProgress("Metal", owner="ops-team", issue="GH-1234")
    override fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        validateMatmulShapes(a.shape, b.shape)
        val resultShape = calculateMatmulShape(a.shape, b.shape)
        val resultData = dataFactory.zeros<T, V>(resultShape, a.dtype)
        return VoidOpsTensor(resultData, a.dtype)
    }

    @InProgress("Metal", owner="ops-team", issue="GH-1234")
    override fun <T : DType, V> transpose(tensor: Tensor<T, V>): Tensor<T, V> {
        val resultShape = calculateTransposeShape(tensor.shape)
        val resultData = dataFactory.zeros<T, V>(resultShape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> permute(tensor: Tensor<T, V>, axes: IntArray): Tensor<T, V> {
        validatePermuteAxes(tensor.shape, axes)
        val resultShape = calculatePermuteShape(tensor.shape, axes)
        val resultData = dataFactory.zeros<T, V>(resultShape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> conv1d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Int,
        padding: Int,
        dilation: Int,
        groups: Int
    ): Tensor<T, V> {
        val resultShape = calculateConv1dShape(input.shape, weight.shape, stride, padding, dilation)
        val resultData = dataFactory.zeros<T, V>(resultShape, input.dtype)
        return VoidOpsTensor(resultData, input.dtype)
    }

    override fun <T : DType, V> conv2d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>,
        groups: Int
    ): Tensor<T, V> {
        val resultShape = calculateConv2dShape(input.shape, weight.shape, stride, padding, dilation)
        val resultData = dataFactory.zeros<T, V>(resultShape, input.dtype)
        return VoidOpsTensor(resultData, input.dtype)
    }

    override fun <T : DType, V> conv3d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Triple<Int, Int, Int>,
        padding: Triple<Int, Int, Int>,
        dilation: Triple<Int, Int, Int>,
        groups: Int
    ): Tensor<T, V> {
        val resultShape = calculateConv3dShape(input.shape, weight.shape, stride, padding, dilation)
        val resultData = dataFactory.zeros<T, V>(resultShape, input.dtype)
        return VoidOpsTensor(resultData, input.dtype)
    }

    override fun <T : DType, V> convTranspose1d(
        input: Tensor<T, V>,
        weight: Tensor<T, V>,
        bias: Tensor<T, V>?,
        stride: Int,
        padding: Int,
        outputPadding: Int,
        dilation: Int,
        groups: Int
    ): Tensor<T, V> {
        val resultShape = calculateConvTranspose1dShape(input.shape, weight.shape, stride, padding, outputPadding, dilation)
        val resultData = dataFactory.zeros<T, V>(resultShape, input.dtype)
        return VoidOpsTensor(resultData, input.dtype)
    }

    override fun <T : DType, V> maxPool2d(
        input: Tensor<T, V>,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>
    ): Tensor<T, V> {
        val resultShape = calculateMaxPool2dShape(input.shape, kernelSize, stride, padding)
        val resultData = dataFactory.zeros<T, V>(resultShape, input.dtype)
        return VoidOpsTensor(resultData, input.dtype)
    }

    override fun <T : DType, V> avgPool2d(
        input: Tensor<T, V>,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        countIncludePad: Boolean
    ): Tensor<T, V> {
        // AvgPool2d has same output shape calculation as MaxPool2d
        val resultShape = calculateMaxPool2dShape(input.shape, kernelSize, stride, padding)
        val resultData = dataFactory.zeros<T, V>(resultShape, input.dtype)
        return VoidOpsTensor(resultData, input.dtype)
    }

    override fun <T : DType, V> upsample2d(
        input: Tensor<T, V>,
        scale: Pair<Int, Int>,
        mode: UpsampleMode,
        alignCorners: Boolean
    ): Tensor<T, V> {
        val resultShape = calculateUpsample2dShape(input.shape, scale)
        val resultData = dataFactory.zeros<T, V>(resultShape, input.dtype)
        return VoidOpsTensor(resultData, input.dtype)
    }

    override fun <T : DType, V> reshape(tensor: Tensor<T, V>, newShape: Shape): Tensor<T, V> {
        val resultShape = calculateReshapeTargetShape(tensor.shape, newShape)
        val resultData = dataFactory.zeros<T, V>(resultShape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> flatten(tensor: Tensor<T, V>, startDim: Int, endDim: Int): Tensor<T, V> {
        val resultShape = calculateFlattenShape(tensor.shape, startDim, endDim)
        val resultData = dataFactory.zeros<T, V>(resultShape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> concat(tensors: List<Tensor<T, V>>, dim: Int): Tensor<T, V> {
        val resultShape = calculateConcatShape(tensors.map { it.shape }, dim)
        val resultData = dataFactory.zeros<T, V>(resultShape, tensors.first().dtype)
        return VoidOpsTensor(resultData, tensors.first().dtype)
    }

    override fun <T : DType, V> split(tensor: Tensor<T, V>, splitSize: Int, dim: Int): List<Tensor<T, V>> {
        val resultShapes = calculateSplitShapes(tensor.shape, splitSize, dim)
        return resultShapes.map { shape ->
            val resultData = dataFactory.zeros<T, V>(shape, tensor.dtype)
            VoidOpsTensor(resultData, tensor.dtype)
        }
    }

    override fun <T : DType, V> squeeze(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        val resultShape = calculateSqueezeShape(tensor.shape, dim)
        val resultData = dataFactory.zeros<T, V>(resultShape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> unsqueeze(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> {
        // Preserve underlying data; return a view with an inserted size-1 dimension
        val newData: sk.ainet.lang.tensor.data.TensorData<T, V> = UnsqueezedTensorData(tensor.data, normalizeUnsqueezeDim(tensor.shape, dim))
        return VoidOpsTensor(newData, tensor.dtype)
    }

    override fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V> {
        // Activation functions preserve shape
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> leakyRelu(tensor: Tensor<T, V>, negativeSlope: Float): Tensor<T, V> {
        // LeakyReLU activation function preserves shape
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> elu(tensor: Tensor<T, V>, alpha: Float): Tensor<T, V> {
        // ELU activation function preserves shape
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> softmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> {
        validateSoftmaxDim(tensor.shape, dim)
        // Softmax preserves shape
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> logSoftmax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> {
        validateSoftmaxDim(tensor.shape, dim)
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> sigmoid(tensor: Tensor<T, V>): Tensor<T, V> {
        // Activation functions preserve shape
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        val resultShape = calculateReductionShape(tensor.shape, dim, "sum")
        val resultData = dataFactory.zeros<T, V>(resultShape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> mean(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        val resultShape = calculateReductionShape(tensor.shape, dim, "mean")
        val resultData = dataFactory.zeros<T, V>(resultShape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <TFrom : DType, TTo : DType, V> convert(
        tensor: Tensor<TFrom, V>,
        targetType: TTo
    ): Tensor<TTo, V> {
        // Type conversion preserves shape but changes dtype
        @Suppress("UNCHECKED_CAST")
        val targetClass = targetType::class as kotlin.reflect.KClass<TTo>
        val resultData = dataFactory.zeros<TTo, V>(tensor.shape, targetClass)
        return VoidOpsTensor(resultData, targetClass)
    }

    override fun <T : DType, V> silu(tensor: Tensor<T, V>): Tensor<T, V> {
        // SiLU (Swish) activation function preserves shape
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> gelu(tensor: Tensor<T, V>): Tensor<T, V> {
        // GELU activation function preserves shape
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> variance(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        val resultShape = calculateReductionShape(tensor.shape, dim, "variance")
        val resultData = dataFactory.zeros<T, V>(resultShape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> argMax(tensor: Tensor<T, V>, dim: Int): Tensor<T, V> {
        // Reduced shape (dim removed) with an Int32 index dtype — mirrors `convert`'s
        // dtype-changing pattern (the traced graph node then carries i32 output).
        val resultShape = calculateReductionShape(tensor.shape, dim, "argMax")
        @Suppress("UNCHECKED_CAST")
        val int32 = sk.ainet.lang.types.Int32::class as kotlin.reflect.KClass<T>
        val resultData = dataFactory.zeros<T, V>(resultShape, int32)
        return VoidOpsTensor(resultData, int32)
    }

    override fun <T : DType, V> sqrt(tensor: Tensor<T, V>): Tensor<T, V> {
        // Square root function preserves shape
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> pow(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        // Power preserves shape (broadcast assumed compatible).
        val resultData = dataFactory.zeros<T, V>(a.shape, a.dtype)
        return VoidOpsTensor(resultData, a.dtype)
    }

    override fun <T : DType, V> powScalar(a: Tensor<T, V>, n: Number): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(a.shape, a.dtype)
        return VoidOpsTensor(resultData, a.dtype)
    }

    override fun <T : DType, V> log(tensor: Tensor<T, V>): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> log2(tensor: Tensor<T, V>): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> log10(tensor: Tensor<T, V>): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> abs(tensor: Tensor<T, V>): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> sign(tensor: Tensor<T, V>): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> clamp(tensor: Tensor<T, V>, minVal: Float, maxVal: Float): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> lt(tensor: Tensor<T, V>, value: Float): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> ge(tensor: Tensor<T, V>, value: Float): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> narrow(tensor: Tensor<T, V>, dim: Int, start: Int, length: Int): Tensor<T, V> {
        val actualDim = if (dim < 0) tensor.shape.rank + dim else dim
        require(actualDim in 0 until tensor.shape.rank) { "narrow dim $dim out of bounds for rank ${tensor.shape.rank}" }
        require(start >= 0 && start + length <= tensor.shape.dimensions[actualDim]) {
            "narrow: start=$start length=$length exceeds dim size ${tensor.shape.dimensions[actualDim]}"
        }
        val resultDims = tensor.shape.dimensions.copyOf()
        resultDims[actualDim] = length
        val resultData = dataFactory.zeros<T, V>(Shape(resultDims), tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> pad2d(tensor: Tensor<T, V>, padLeft: Int, padRight: Int, padTop: Int, padBottom: Int): Tensor<T, V> {
        require(tensor.shape.rank == 4) { "pad2d requires 4D tensor [N,C,H,W], got rank ${tensor.shape.rank}" }
        val dims = tensor.shape.dimensions.copyOf()
        dims[2] = dims[2] + padTop + padBottom
        dims[3] = dims[3] + padLeft + padRight
        val resultData = dataFactory.zeros<T, V>(Shape(dims), tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

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
        val resultData = dataFactory.zeros<T, V>(Shape(resultDims), tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> tril(tensor: Tensor<T, V>, k: Int): Tensor<T, V> {
        // tril preserves shape
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> gather(input: Tensor<T, V>, indices: Tensor<DType, *>, dim: Int): Tensor<T, V> {
        // Gather selects rows along `dim`, replacing that axis with the FULL
        // indices shape (not just its first dim). Matches DefaultCpuOps.gather:
        // for a [vocab, emb] table and [batch, seq] indices the result is
        // [batch, seq, emb]. The previous `resultDims[dim] = indices.shape[0]`
        // collapsed multi-dim indices to a single row, breaking the downstream
        // reshape (e.g. embedding lookup) during weight-free tracing.
        val inDims = input.shape.dimensions
        val idxDims = indices.shape.dimensions
        val resultDims = inDims.copyOfRange(0, dim) + idxDims + inDims.copyOfRange(dim + 1, inDims.size)
        val resultData = dataFactory.zeros<T, V>(Shape(resultDims), input.dtype)
        return VoidOpsTensor(resultData, input.dtype)
    }

    override fun <T : DType, V> indexSelect(input: Tensor<T, V>, indices: Tensor<DType, *>, dim: Int): Tensor<T, V> {
        val resultDims = input.shape.dimensions.copyOf()
        resultDims[dim] = indices.shape.dimensions[0]
        val resultData = dataFactory.zeros<T, V>(Shape(resultDims), input.dtype)
        return VoidOpsTensor(resultData, input.dtype)
    }

    override fun <T : DType, V> exp(tensor: Tensor<T, V>): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> expm1(tensor: Tensor<T, V>): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> sin(tensor: Tensor<T, V>): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> cos(tensor: Tensor<T, V>): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> tanh(tensor: Tensor<T, V>): Tensor<T, V> {
        val resultData = dataFactory.zeros<T, V>(tensor.shape, tensor.dtype)
        return VoidOpsTensor(resultData, tensor.dtype)
    }

    override fun <T : DType, V> scaledDotProductAttention(
        query: Tensor<T, V>,
        key: Tensor<T, V>,
        value: Tensor<T, V>,
        mask: Tensor<T, V>?,
        scale: Float,
        causal: Boolean
    ): Tensor<T, V> {
        // Output shape matches query shape: [batch, nHeads, seqLen, headDim]
        val resultData = dataFactory.zeros<T, V>(query.shape, query.dtype)
        return VoidOpsTensor(resultData, query.dtype)
    }

    /**
     * Validates shapes for matrix multiplication.
     * For 2D matrices: (m, k) × (k, n) -> (m, n)
     * For higher dimensions: batch dimensions must match, inner dimensions must be compatible
     */
    private fun validateMatmulShapes(a: Shape, b: Shape) {
        // Support PyTorch-like matmul rank rules, including 1D operands via virtual unsqueeze
        if (a.rank == 0 || b.rank == 0) {
            throw IllegalArgumentException("Matrix multiplication requires tensors with at least 1 dimension per operand")
        }

        // Build effective shapes by virtually unsqueezing 1D operands:
        // - If a is 1D with dim n, treat as (1, n)
        // - If b is 1D with dim n, treat as (n, 1)
        val aEffDims = when (a.rank) {
            1 -> intArrayOf(1, a.dimensions[0])
            else -> a.dimensions
        }
        val bEffDims = when (b.rank) {
            1 -> intArrayOf(b.dimensions[0], 1)
            else -> b.dimensions
        }

        val aEffRank = aEffDims.size
        val bEffRank = bEffDims.size

        // Inner dimension (k) must match: a[..., k] with b[k, ...]
        val aK = aEffDims[aEffRank - 1]
        val bK = bEffDims[bEffRank - 2]
        if (aK != bK) {
            throw IllegalArgumentException(
                "Matrix multiplication shape mismatch: inner dimensions must match ($aK vs $bK)"
            )
        }

        // Validate broadcastability of leading batch dims (all but the last 2 dims)
        val aBatchRank = aEffRank - 2
        val bBatchRank = bEffRank - 2
        val maxBatchRank = maxOf(aBatchRank, bBatchRank)
        for (i in 0 until maxBatchRank) {
            val aDim = if (i < aBatchRank) aEffDims[i] else 1
            val bDim = if (i < bBatchRank) bEffDims[i] else 1
            if (aDim != bDim && aDim != 1 && bDim != 1) {
                throw IllegalArgumentException(
                    "Matrix multiplication batch dimension mismatch at position $i: $aDim vs $bDim"
                )
            }
        }
    }

    /**
     * Calculates the result shape for matrix multiplication
     */
    private fun calculateMatmulShape(a: Shape, b: Shape): Shape {
        // Construct effective shapes by virtually unsqueezing 1D operands
        val aEff = when (a.rank) {
            1 -> intArrayOf(1, a.dimensions[0])
            else -> a.dimensions
        }
        val bEff = when (b.rank) {
            1 -> intArrayOf(b.dimensions[0], 1)
            else -> b.dimensions
        }
        val aEffRank = aEff.size
        val bEffRank = bEff.size

        // Compute broadcasted batch dims
        val batchRank = maxOf(aEffRank, bEffRank) - 2
        val outBatch = IntArray(batchRank) { i ->
            val aDim = if (i < aEffRank - 2) aEff[i] else 1
            val bDim = if (i < bEffRank - 2) bEff[i] else 1
            maxOf(aDim, bDim)
        }

        val m = aEff[aEffRank - 2]
        val n = bEff[bEffRank - 1]

        // Build full result then squeeze depending on original ranks
        return when {
            a.rank == 1 && b.rank == 1 -> {
                // Dot product -> scalar
                Shape(intArrayOf())
            }
            a.rank == 1 -> {
                // (k,) @ (..., k, n) -> (..., n)
                val result = IntArray(outBatch.size + 1)
                for (i in outBatch.indices) result[i] = outBatch[i]
                result[result.size - 1] = n
                Shape(result)
            }
            b.rank == 1 -> {
                // (..., m, k) @ (k,) -> (..., m)
                val result = IntArray(outBatch.size + 1)
                for (i in outBatch.indices) result[i] = outBatch[i]
                result[result.size - 1] = m
                Shape(result)
            }
            else -> {
                // Regular case: (..., m, k) @ (..., k, n) -> (..., m, n)
                val result = IntArray(outBatch.size + 2)
                for (i in outBatch.indices) result[i] = outBatch[i]
                result[result.size - 2] = m
                result[result.size - 1] = n
                Shape(result)
            }
        }
    }

    /**
     * Calculates the result shape for transpose operation.
     * For 2D tensors: (m, n) -> (n, m)
     * For higher dimensions: swaps the last two dimensions
     */
    /**
     * Validate that [axes] is a valid permutation of `0..shape.rank-1`.
     */
    internal fun validatePermuteAxes(shape: Shape, axes: IntArray) {
        require(axes.size == shape.rank) {
            "permute: axes length ${axes.size} must match tensor rank ${shape.rank}"
        }
        val seen = BooleanArray(shape.rank)
        for (a in axes) {
            require(a in 0 until shape.rank) {
                "permute: axis $a out of range [0, ${shape.rank})"
            }
            require(!seen[a]) { "permute: axis $a appears more than once in $axes" }
            seen[a] = true
        }
    }

    /**
     * Result shape after applying [axes] permutation to [shape].
     */
    internal fun calculatePermuteShape(shape: Shape, axes: IntArray): Shape {
        val dims = IntArray(shape.rank) { i -> shape.dimensions[axes[i]] }
        return Shape(dims)
    }

    private fun calculateTransposeShape(shape: Shape): Shape {
        if (shape.rank < 2) {
            throw IllegalArgumentException("Transpose requires tensors with at least 2 dimensions")
        }
        
        val resultDims = shape.dimensions.copyOf()
        val lastIdx = resultDims.size - 1
        val secondLastIdx = resultDims.size - 2
        
        // Swap last two dimensions
        val temp = resultDims[lastIdx]
        resultDims[lastIdx] = resultDims[secondLastIdx]
        resultDims[secondLastIdx] = temp
        
        return Shape(resultDims)
    }

    /**
     * Calculates target shape for reshape, supporting a single -1 dimension for inference.
     * Validates that total volume remains the same and no illegal dimensions are provided.
     */
    private fun calculateReshapeTargetShape(originalShape: Shape, target: Shape): Shape {
        val dims = target.dimensions
        // A dynamic extent (either in the target or the input) passes through unchanged: volume-based
        // inference/validation is undefined when an extent is unknown. This is distinct from `-1` = infer,
        // which the distinct [Dim.DYNAMIC] sentinel keeps separate.
        if (dims.hasDynamic() || originalShape.dimensions.hasDynamic()) {
            require(dims.none { it == -1 }) {
                "reshape cannot infer a `-1` dimension while the shape is dynamic: ${dims.toList()}"
            }
            return Shape(dims.copyOf())
        }
        val total = originalShape.volume
        var inferIndex = -1
        var product = 1
        for ((i, d) in dims.withIndex()) {
            when {
                d == -1 -> {
                    if (inferIndex != -1) {
                        throw IllegalArgumentException("Only one dimension can be -1 in reshape, found at $inferIndex and $i")
                    }
                    inferIndex = i
                }
                d < -1 -> {
                    throw IllegalArgumentException("Reshape dimensions must be non-negative or -1 for inference, got $d at index $i")
                }
                else -> {
                    product *= d
                }
            }
        }
        return if (inferIndex >= 0) {
            if (product == 0) {
                // If any explicit dim is 0, the only consistent inference for zero total is 0
                if (total != 0) {
                    throw IllegalArgumentException("Reshape volume mismatch: original volume $total != new volume $product")
                }
                val out = dims.copyOf()
                out[inferIndex] = 0
                Shape(out)
            } else {
                if (total % product != 0) {
                    throw IllegalArgumentException("Cannot infer reshape dimension: original volume $total is not divisible by specified product $product")
                }
                val inferred = total / product
                val out = dims.copyOf()
                out[inferIndex] = inferred
                Shape(out)
            }
        } else {
            if (product != total) {
                throw IllegalArgumentException("Reshape volume mismatch: original volume $total != new volume $product")
            }
            Shape(dims.copyOf())
        }
    }

    /**
     * Calculates the result shape for flatten operation
     */
    private fun calculateFlattenShape(shape: Shape, startDim: Int, endDim: Int): Shape {
        val actualStartDim = if (startDim < 0) shape.rank + startDim else startDim
        val actualEndDim = if (endDim < 0) shape.rank + endDim else endDim
        
        if (actualStartDim < 0 || actualStartDim >= shape.rank) {
            throw IllegalArgumentException("Start dimension $startDim is out of bounds for tensor with ${shape.rank} dimensions")
        }
        if (actualEndDim < 0 || actualEndDim >= shape.rank) {
            throw IllegalArgumentException("End dimension $endDim is out of bounds for tensor with ${shape.rank} dimensions")
        }
        if (actualStartDim > actualEndDim) {
            throw IllegalArgumentException("Start dimension $actualStartDim must be <= end dimension $actualEndDim")
        }
        
        val resultDims = mutableListOf<Int>()
        
        // Add dimensions before startDim
        for (i in 0 until actualStartDim) {
            resultDims.add(shape.dimensions[i])
        }
        
        // Calculate flattened dimension
        var flattenedSize = 1
        for (i in actualStartDim..actualEndDim) {
            flattenedSize *= shape.dimensions[i]
        }
        resultDims.add(flattenedSize)
        
        // Add dimensions after endDim
        for (i in actualEndDim + 1 until shape.rank) {
            resultDims.add(shape.dimensions[i])
        }
        
        return Shape(resultDims.toIntArray())
    }

    /**
     * Validates softmax dimension parameter
     */
    private fun validateSoftmaxDim(shape: Shape, dim: Int) {
        val actualDim = if (dim < 0) shape.rank + dim else dim
        if (actualDim < 0 || actualDim >= shape.rank) {
            throw IllegalArgumentException("Softmax dimension $dim is out of bounds for tensor with ${shape.rank} dimensions")
        }
    }

    /**
     * Calculates the result shape for reduction operations (sum, mean)
     */
    private fun calculateReductionShape(shape: Shape, dim: Int?, operation: String): Shape {
        return if (dim == null) {
            // Reduce all dimensions to scalar
            Shape(1)
        } else {
            val actualDim = if (dim < 0) shape.rank + dim else dim
            if (actualDim < 0 || actualDim >= shape.rank) {
                throw IllegalArgumentException("$operation dimension $dim is out of bounds for tensor with ${shape.rank} dimensions")
            }
            
            // Remove the specified dimension
            val resultDims = shape.dimensions.filterIndexed { index, _ -> index != actualDim }.toIntArray()
            if (resultDims.isEmpty()) {
                Shape(1) // Result is scalar if all dimensions are reduced
            } else {
                Shape(resultDims)
            }
        }
    }

    /**
     * Calculates the result shape for conv1d operation.
     * Input shape: (batch, in_channels, length)
     * Weight shape: (out_channels, in_channels_per_group, kernel_length)
     * Output shape: (batch, out_channels, out_length)
     */
    private fun calculateConv1dShape(
        inputShape: Shape,
        weightShape: Shape,
        stride: Int,
        padding: Int,
        dilation: Int
    ): Shape = Shape(
        ConvShapeUtils.conv1dOutputShape(
            inputShape.dimensions, weightShape.dimensions, stride, padding, dilation
        )
    )

    /**
     * Calculates the result shape for conv3d operation.
     * Input shape: (batch, in_channels, depth, height, width)
     * Weight shape: (out_channels, in_channels_per_group, kernel_depth, kernel_height, kernel_width)
     * Output shape: (batch, out_channels, out_depth, out_height, out_width)
     */
    private fun calculateConv3dShape(
        inputShape: Shape,
        weightShape: Shape,
        stride: Triple<Int, Int, Int>,
        padding: Triple<Int, Int, Int>,
        dilation: Triple<Int, Int, Int>
    ): Shape = Shape(
        ConvShapeUtils.conv3dOutputShape(
            inputShape.dimensions, weightShape.dimensions, stride, padding, dilation
        )
    )

    /**
     * Calculates the result shape for convTranspose1d operation.
     * Input shape: (batch, in_channels, length)
     * Weight shape: (in_channels, out_channels_per_group, kernel_size)
     * Output shape: (batch, out_channels, out_length)
     */
    private fun calculateConvTranspose1dShape(
        inputShape: Shape, weightShape: Shape, stride: Int, padding: Int, outputPadding: Int, dilation: Int
    ): Shape {
        val batch = inputShape[0]
        val outChannels = weightShape[1]
        val inputLength = inputShape[2]
        val kernelSize = weightShape[2]
        val outputLength = (inputLength - 1) * stride - 2 * padding + dilation * (kernelSize - 1) + outputPadding + 1
        return Shape(batch, outChannels, outputLength)
    }

    /**
     * Calculates the result shape for conv2d operation.
     * Input shape: (batch, in_channels, height, width)
     * Weight shape: (out_channels, in_channels_per_group, kernel_height, kernel_width)
     * Output shape: (batch, out_channels, out_height, out_width)
     */
    private fun calculateConv2dShape(
        inputShape: Shape,
        weightShape: Shape,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>,
        dilation: Pair<Int, Int>
    ): Shape = Shape(
        ConvShapeUtils.conv2dOutputShape(
            inputShape.dimensions, weightShape.dimensions, stride, padding, dilation
        )
    )

    /**
     * Calculates the result shape for maxPool2d operation.
     * Input shape: (batch, channels, height, width)
     * Output shape: (batch, channels, out_height, out_width)
     */
    private fun calculateMaxPool2dShape(
        inputShape: Shape,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
        padding: Pair<Int, Int>
    ): Shape {
        if (inputShape.rank != 4) {
            throw IllegalArgumentException("MaxPool2d input must be 4D tensor (batch, channels, height, width)")
        }
        
        val batch = inputShape.dimensions[0]
        val channels = inputShape.dimensions[1]
        val inputHeight = inputShape.dimensions[2]
        val inputWidth = inputShape.dimensions[3]
        
        val (kernelH, kernelW) = kernelSize
        val (strideH, strideW) = stride
        val (padH, padW) = padding
        
        val outputHeight = ((inputHeight + 2 * padH - kernelH) / strideH) + 1
        val outputWidth = ((inputWidth + 2 * padW - kernelW) / strideW) + 1
        
        return Shape(batch, channels, outputHeight, outputWidth)
    }

    /**
     * Calculates the result shape for concat operation
     */
    private fun calculateConcatShape(shapes: List<Shape>, dim: Int): Shape {
        if (shapes.isEmpty()) {
            throw IllegalArgumentException("Cannot concatenate empty list of tensors")
        }
        
        val firstShape = shapes.first()
        val actualDim = if (dim < 0) firstShape.rank + dim else dim
        
        if (actualDim < 0 || actualDim >= firstShape.rank) {
            throw IllegalArgumentException("Concatenation dimension $dim is out of bounds for tensor with ${firstShape.rank} dimensions")
        }
        // Validate all shapes are compatible (same except in concat dimension)
        for (shape in shapes.drop(1)) {
            if (shape.rank != firstShape.rank) {
                throw IllegalArgumentException("All tensors must have the same number of dimensions for concatenation")
            }
            for (i in shape.dimensions.indices) {
                // Off-axis extents must match; a dynamic extent is compatible with any concrete size.
                if (i != actualDim && !Dim.compatible(shape.dimensions[i], firstShape.dimensions[i])) {
                    throw IllegalArgumentException(
                        "All tensors must have the same shape except in the concatenation dimension. " +
                        "Dimension $i: ${firstShape.dimensions[i]} vs ${shape.dimensions[i]}"
                    )
                }
            }
        }
        
        // Calculate result shape. [Dim.concat] keeps the concatenated axis dynamic when any input is
        // dynamic there (a growing KV cache `? ++ 1` stays `?`) instead of numerically summing it, which
        // would corrupt the shape — exactly what blocked threading a real dynamic seq dim through a decode
        // trace.
        val resultDims = firstShape.dimensions.copyOf()
        resultDims[actualDim] = Dim.concat(shapes.map { it.dimensions[actualDim] })

        return Shape(resultDims)
    }

    /**
     * Calculates the result shapes for split operation
     */
    private fun calculateSplitShapes(shape: Shape, splitSize: Int, dim: Int): List<Shape> {
        val actualDim = if (dim < 0) shape.rank + dim else dim
        
        if (actualDim < 0 || actualDim >= shape.rank) {
            throw IllegalArgumentException("Split dimension $dim is out of bounds for tensor with ${shape.rank} dimensions")
        }
        
        if (splitSize <= 0) {
            throw IllegalArgumentException("Split size must be positive, got $splitSize")
        }
        
        val dimSize = shape.dimensions[actualDim]
        require(splitSize > 0) { "Split size must be positive, got $splitSize" }
        val remainder = dimSize % splitSize
        if (remainder != 0) {
            throw IllegalArgumentException("Dimension $actualDim size $dimSize is not divisible by split size $splitSize")
        }
        val fullSplits = dimSize / splitSize
        val result = MutableList(fullSplits) {
            val dims = shape.dimensions.copyOf()
            dims[actualDim] = splitSize
            Shape(dims)
        }
        return result
    }

    /**
     * Calculates the result shape for 2D upsampling.
     */
    private fun calculateUpsample2dShape(shape: Shape, scale: Pair<Int, Int>): Shape {
        require(shape.rank == 4) {
            "Upsample2d expects 4D input (N, C, H, W), got ${shape.dimensions.contentToString()}"
        }
        val (scaleH, scaleW) = scale
        require(scaleH > 0 && scaleW > 0) { "Upsample2d scale factors must be positive" }
        val dims = shape.dimensions.copyOf()
        dims[2] = dims[2] * scaleH
        dims[3] = dims[3] * scaleW
        return Shape(dims)
    }

    /**
     * Calculates the result shape for squeeze operation
     */
    private fun calculateSqueezeShape(shape: Shape, dim: Int?): Shape {
        return if (dim == null) {
            // Remove all dimensions of size 1
            val resultDims = shape.dimensions.filter { it != 1 }.toIntArray()
            if (resultDims.isEmpty()) {
                Shape(1) // If all dimensions were 1, result is scalar
            } else {
                Shape(resultDims)
            }
        } else {
            val actualDim = if (dim < 0) shape.rank + dim else dim
            
            if (actualDim < 0 || actualDim >= shape.rank) {
                throw IllegalArgumentException("Squeeze dimension $dim is out of bounds for tensor with ${shape.rank} dimensions")
            }
            
            if (shape.dimensions[actualDim] != 1) {
                throw IllegalArgumentException(
                    "Cannot squeeze dimension $actualDim with size ${shape.dimensions[actualDim]}. Only dimensions of size 1 can be squeezed."
                )
            }
            
            // Remove the specified dimension
            val resultDims = shape.dimensions.filterIndexed { index, _ -> index != actualDim }.toIntArray()
            if (resultDims.isEmpty()) {
                Shape(1) // Result is scalar if all dimensions are removed
            } else {
                Shape(resultDims)
            }
        }
    }

    /**
     * Calculates the result shape for unsqueeze operation
     */
    private fun calculateUnsqueezeShape(shape: Shape, dim: Int): Shape {
        val newRank = shape.rank + 1
        val actualDim = if (dim < 0) newRank + dim else dim
        
        if (actualDim < 0 || actualDim >= newRank) {
            throw IllegalArgumentException("Unsqueeze dimension $dim is out of bounds for new tensor with $newRank dimensions")
        }
        
        val resultDims = IntArray(newRank)
        var originalIndex = 0
        
        for (i in 0 until newRank) {
            if (i == actualDim) {
                resultDims[i] = 1
            } else {
                resultDims[i] = shape.dimensions[originalIndex]
                originalIndex++
            }
        }
        
        return Shape(resultDims)
    }

    /**
     * Normalizes an unsqueeze dim possibly negative to the actual index after insertion.
     */
    private fun normalizeUnsqueezeDim(shape: Shape, dim: Int): Int {
        val newRank = shape.rank + 1
        val actualDim = if (dim < 0) newRank + dim else dim
        if (actualDim < 0 || actualDim >= newRank) {
            throw IllegalArgumentException("Unsqueeze dimension $dim is out of bounds for new tensor with $newRank dimensions")
        }
        return actualDim
    }
}

/**
 * A [TensorData] that carries only a [Shape] and allocates NO backing buffer. Used by
 * [VoidTensorOps] for shape-only tracing: element access is never valid (nothing to read/write),
 * but crucially the shape may contain a dynamic extent (`-1`) that a real allocation would reject.
 */
private class ShapeOnlyTensorData<T : DType, V>(override val shape: Shape) : TensorData<T, V> {
    private fun noData(): Nothing =
        error("shape-only (void) tensor carries no data — tracing propagates shapes only")
    override fun get(vararg indices: Int): V = noData()
    override fun set(vararg indices: Int, value: V): Unit = noData()
    override fun copyToFloatArray(): FloatArray = noData()
}

/** Drop-in for the one `dataFactory.zeros` call VoidTensorOps makes. A static shape delegates to
 *  [DenseTensorDataFactory] (real, readable zeros — preserves existing behavior); only a DYNAMIC shape,
 *  which cannot be allocated, gets the allocation-free [ShapeOnlyTensorData] so its `-1` extent survives. */
private object ShapeOnlyDataFactory {
    private val dense = DenseTensorDataFactory()
    fun <T : DType, V> zeros(shape: Shape, dtype: KClass<T>): TensorData<T, V> =
        if (shape.hasDynamic()) ShapeOnlyTensorData(shape) else dense.zeros(shape, dtype)
}
