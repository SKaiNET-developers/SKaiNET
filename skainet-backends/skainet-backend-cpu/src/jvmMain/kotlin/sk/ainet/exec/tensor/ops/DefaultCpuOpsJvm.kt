package sk.ainet.exec.tensor.ops

import jdk.incubator.vector.FloatVector
import jdk.incubator.vector.VectorSpecies
import jdk.incubator.vector.VectorOperators
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.Q8_0TensorData
import sk.ainet.lang.tensor.data.Q4_KTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.data.TensorDataFactory
import kotlin.math.max

internal class DefaultCpuOpsJvm(
    dataFactory: TensorDataFactory,
) : DefaultCpuOpsBase(dataFactory) {

    private val floatSpecies: VectorSpecies<Float> = FloatVector.SPECIES_PREFERRED

    override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        vectorFloatBinary(a, b, { x, y -> x.add(y) }) { x, y -> x + y }?.let { return it }
        return super.add(a, b)
    }

    override fun <T : DType, V> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        vectorFloatBinary(a, b, { x, y -> x.sub(y) }) { x, y -> x - y }?.let { return it }
        return super.subtract(a, b)
    }

    override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        vectorFloatBinary(a, b, { x, y -> x.mul(y) }) { x, y -> x * y }?.let { return it }
        return super.multiply(a, b)
    }

    override fun <T : DType, V> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        vectorFloatBinary(a, b, { x, y -> x.div(y) }) { x, y -> x / y }?.let { return it }
        return super.divide(a, b)
    }

    override fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        // Try quantized matmul path first (FP32 input x quantized weights)
        chooseQuantizedMatmul(a, b)?.let { return it }
        // Fallback to standard FP32 matmul
        chooseMatmul(a, b)?.let { return it }
        return super.matmul(a, b)
    }

    /**
     * Dispatch to vectorized quantized matmul kernels for Q8_0 and Q4_K weights.
     * Input must be FP32, weights can be Q8_0TensorData or Q4_KTensorData.
     */
    private fun <T : DType, V> chooseQuantizedMatmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>? {
        // Input must be FP32 with FloatArray backing
        if (a.dtype != FP32::class) return null
        val aData = a.data as? FloatArrayTensorData<*> ?: return null
        if (a.shape.rank != 2) return null

        val bData = b.data
        val bShape = b.shape
        if (bShape.rank != 2) return null

        val batchSize = a.shape[0]
        val inputDim = a.shape[1]
        val weightInputDim = bShape[0]
        val outputDim = bShape[1]

        if (inputDim != weightInputDim) return null

        return when (bData) {
            is Q8_0TensorData -> {
                val outBuffer = FloatArray(batchSize * outputDim)
                for (batch in 0 until batchSize) {
                    val inputOffset = batch * inputDim
                    val outputOffset = batch * outputDim
                    JvmQuantizedVectorKernels.matmulQ8_0Vec(
                        aData.buffer,
                        bData.packedData,
                        inputDim,
                        outputDim,
                        outBuffer,
                        outputOffset
                    )
                }
                val outData = DenseFloatArrayTensorData<T>(Shape(batchSize, outputDim), outBuffer)
                @Suppress("UNCHECKED_CAST")
                CpuTensor(outData as TensorData<T, V>, this, a.dtype)
            }
            is Q4_KTensorData -> {
                val outBuffer = FloatArray(batchSize * outputDim)
                for (batch in 0 until batchSize) {
                    val inputOffset = batch * inputDim
                    val outputOffset = batch * outputDim
                    JvmQuantizedVectorKernels.matmulQ4_KVec(
                        aData.buffer,
                        bData.packedData,
                        inputDim,
                        outputDim,
                        outBuffer,
                        outputOffset
                    )
                }
                val outData = DenseFloatArrayTensorData<T>(Shape(batchSize, outputDim), outBuffer)
                @Suppress("UNCHECKED_CAST")
                CpuTensor(outData as TensorData<T, V>, this, a.dtype)
            }
            else -> null
        }
    }

    override fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V> {
        vectorFloatUnary(tensor, { vector ->
            val zero = FloatVector.zero(floatSpecies)
            vector.max(zero)
        }, { value ->
            if (value < 0f) 0f else value
        })?.let { return it }
        return super.relu(tensor)
    }

    override fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        if (dim == null) {
            vectorFloatReduceAllSum<T, V>(tensor)?.let { return it }
        }
        return super.sum(tensor, dim)
    }

    override fun <T : DType, V> mean(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        if (dim == null) {
            val volume = tensor.shape.volume
            if (volume == 0) return super.mean(tensor, dim)
            vectorFloatReduceAllSum<T, V>(tensor)?.let { reduced ->
                val out = reduced as CpuTensor<T, V>
                @Suppress("UNCHECKED_CAST")
                val floatData = out.data as DenseFloatArrayTensorData<T>
                floatData.buffer[0] = floatData.buffer[0] / volume.toFloat()
                return reduced
            }
        }
        return super.mean(tensor, dim)
    }

    private fun <T : DType, V> vectorFloatBinary(
        a: Tensor<T, V>,
        b: Tensor<T, V>,
        vectorOp: (FloatVector, FloatVector) -> FloatVector,
        scalarOp: (Float, Float) -> Float
    ): Tensor<T, V>? {
        // Only FP32/FP16 supported for vector path
        if (!supportsFloatOps(a) || !supportsFloatOps(b)) return null
        if (a.dtype != b.dtype) return null

        val aData = a.data as? FloatArrayTensorData<T> ?: return null
        val bData = b.data as? FloatArrayTensorData<T> ?: return null

        // Determine broadcasted output shape
        val outShape = try { broadcastShapes(a.shape, b.shape) } catch (e: IllegalArgumentException) { return null }
        val outVolume = outShape.volume
        val outBuffer = FloatArray(outVolume)

        val aVol = a.shape.volume
        val bVol = b.shape.volume

        // Case 1: exact shape match (fast path)
        if (a.shape == b.shape) {
            JvmVectorKernels.binaryFloat(aData.buffer, bData.buffer, outBuffer, outVolume, vectorOp, scalarOp)
            val outData = DenseFloatArrayTensorData<T>(Shape(outShape.dimensions.copyOf()), outBuffer)
            @Suppress("UNCHECKED_CAST")
            return CpuTensor(outData as TensorData<T, V>, this, a.dtype)
        }

        // Case 2: scalar broadcast
        if (aVol == 1) {
            val aval = aData.buffer[0]
            val speciesLen = floatSpecies.length()
            var idx = 0
            val loopBound = floatSpecies.loopBound(outVolume)
            while (idx < loopBound) {
                val va = FloatVector.broadcast(floatSpecies, aval)
                val vb = FloatVector.fromArray(floatSpecies, bData.buffer, idx)
                vectorOp(va, vb).intoArray(outBuffer, idx)
                idx += speciesLen
            }
            while (idx < outVolume) {
                outBuffer[idx] = scalarOp(aval, bData.buffer[idx])
                idx++
            }
            val outData = DenseFloatArrayTensorData<T>(Shape(outShape.dimensions.copyOf()), outBuffer)
            @Suppress("UNCHECKED_CAST")
            return CpuTensor(outData as TensorData<T, V>, this, a.dtype)
        }
        if (bVol == 1) {
            val bval = bData.buffer[0]
            val speciesLen = floatSpecies.length()
            var idx = 0
            val loopBound = floatSpecies.loopBound(outVolume)
            while (idx < loopBound) {
                val va = FloatVector.fromArray(floatSpecies, aData.buffer, idx)
                val vb = FloatVector.broadcast(floatSpecies, bval)
                vectorOp(va, vb).intoArray(outBuffer, idx)
                idx += speciesLen
            }
            while (idx < outVolume) {
                outBuffer[idx] = scalarOp(aData.buffer[idx], bval)
                idx++
            }
            val outData = DenseFloatArrayTensorData<T>(Shape(outShape.dimensions.copyOf()), outBuffer)
            @Suppress("UNCHECKED_CAST")
            return CpuTensor(outData as TensorData<T, V>, this, a.dtype)
        }

        // Case 3: last-dimension broadcasting (bias add). Supports arbitrary leading dims.
        val aLast = if (a.shape.rank > 0) a.shape[a.shape.rank - 1] else 1
        val bLast = if (b.shape.rank > 0) b.shape[b.shape.rank - 1] else 1
        val outLast = outShape.dimensions.lastOrNull() ?: 1
        val groups = if (outLast == 0) 0 else outVolume / outLast
        if (groups > 0) {
            // b broadcasts across leading dims if its last dim == outLast and all other dims are 1 or equal
            val bIsBias = (b.shape.rank == 1 && bLast == outLast) || (
                b.shape.rank >= 1 && bLast == outLast && b.shape.dimensions.dropLast(1).all { it == 1 }
            )
            val aIsBias = (a.shape.rank == 1 && aLast == outLast) || (
                a.shape.rank >= 1 && aLast == outLast && a.shape.dimensions.dropLast(1).all { it == 1 }
            )
            if (bIsBias && aVol == outVolume) {
                val step = floatSpecies.length()
                val loopBoundTail = floatSpecies.loopBound(outLast)
                for (g in 0 until groups) {
                    val aOff = g * outLast
                    var idx = 0
                    while (idx < loopBoundTail) {
                        val va = FloatVector.fromArray(floatSpecies, aData.buffer, aOff + idx)
                        val vb = FloatVector.fromArray(floatSpecies, bData.buffer, idx)
                        vectorOp(va, vb).intoArray(outBuffer, aOff + idx)
                        idx += step
                    }
                    while (idx < outLast) {
                        outBuffer[aOff + idx] = scalarOp(aData.buffer[aOff + idx], bData.buffer[idx])
                        idx++
                    }
                }
                val outData = DenseFloatArrayTensorData<T>(Shape(outShape.dimensions.copyOf()), outBuffer)
                @Suppress("UNCHECKED_CAST")
                return CpuTensor(outData as TensorData<T, V>, this, a.dtype)
            }
            if (aIsBias && bVol == outVolume) {
                val step = floatSpecies.length()
                val loopBoundTail = floatSpecies.loopBound(outLast)
                for (g in 0 until groups) {
                    val bOff = g * outLast
                    var idx = 0
                    while (idx < loopBoundTail) {
                        val va = FloatVector.fromArray(floatSpecies, aData.buffer, idx)
                        val vb = FloatVector.fromArray(floatSpecies, bData.buffer, bOff + idx)
                        vectorOp(va, vb).intoArray(outBuffer, bOff + idx)
                        idx += step
                    }
                    while (idx < outLast) {
                        outBuffer[bOff + idx] = scalarOp(aData.buffer[idx], bData.buffer[bOff + idx])
                        idx++
                    }
                }
                val outData = DenseFloatArrayTensorData<T>(Shape(outShape.dimensions.copyOf()), outBuffer)
                @Suppress("UNCHECKED_CAST")
                return CpuTensor(outData as TensorData<T, V>, this, a.dtype)
            }
        }

        // Fallback when complex broadcasting not supported here
        return null
    }

    private fun <T : DType, V> vectorFloatUnary(
        tensor: Tensor<T, V>,
        vectorOp: (FloatVector) -> FloatVector,
        scalarOp: (Float) -> Float
    ): Tensor<T, V>? {
        if (!supportsFloatOps(tensor)) return null
        val tensorData = tensor.data as? FloatArrayTensorData<T> ?: return null
        val volume = tensor.shape.volume
        val outBuffer = FloatArray(volume)
        JvmVectorKernels.unaryFloat(tensorData.buffer, outBuffer, volume, vectorOp, scalarOp)
        val outData = DenseFloatArrayTensorData<T>(Shape(tensor.shape.dimensions.copyOf()), outBuffer)
        @Suppress("UNCHECKED_CAST")
        return CpuTensor(outData as TensorData<T, V>, this, tensor.dtype)
    }

    private fun <T : DType> supportsFloatOps(a: Tensor<T, *>, b: Tensor<T, *>): Boolean {
        return supportsFloatOps(a) &&
            a.dtype == b.dtype &&
            a.shape == b.shape
    }

    private fun <T : DType> supportsFloatOps(tensor: Tensor<T, *>): Boolean {
        val dtype = tensor.dtype
        return (dtype == FP32::class || dtype == FP16::class)
    }

    private fun <T : DType, V> chooseMatmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V>? {
        if (!supportsFloatOps(a) || !supportsFloatOps(b)) return null
        if (a.dtype != b.dtype) return null
        if (a.shape.rank != 2 || b.shape.rank != 2) return null

        val aRows = a.shape[0]
        val aCols = a.shape[1]
        val bRows = b.shape[0]
        val bCols = b.shape[1]
        if (aCols != bRows) return null

        val aData = a.data as? FloatArrayTensorData<T> ?: return null
        val bData = b.data as? FloatArrayTensorData<T> ?: return null

        // Heuristics
        val m = aRows
        val n = bCols
        val k = aCols
        val work = m.toLong() * n.toLong() * k.toLong()

        val outBuffer = FloatArray(m * n)

        // Try BLAS for large sizes if enabled and available
        if (JvmCpuBackendConfig.blasEnabled && JvmBlas.isAvailable()) {
            val blasThreshold = 512L * 512L * 256L // tuneable
            if (work >= blasThreshold) {
                val ok = JvmBlas.sgemmRowMajorNN(m, n, k, 1f, aData.buffer, bData.buffer, outBuffer)
                if (ok) {
                    val outData = DenseFloatArrayTensorData<T>(Shape(m, n), outBuffer)
                    @Suppress("UNCHECKED_CAST")
                    return CpuTensor(outData as TensorData<T, V>, this, a.dtype)
                }
            }
        }

        // Use blocked matmul for small/medium sizes
        val blockedThreshold = 16 * 16 // always use blocked above tiny cases
        if (m >= blockedThreshold || n >= blockedThreshold || k >= blockedThreshold) {
            JvmVectorKernels.matmulFloatBlocked(m, k, n, aData.buffer, bData.buffer, outBuffer)
            val outData = DenseFloatArrayTensorData<T>(Shape(m, n), outBuffer)
            @Suppress("UNCHECKED_CAST")
            return CpuTensor(outData as TensorData<T, V>, this, a.dtype)
        }

        // Fallback to simple vectorized inner-product matmul
        JvmVectorKernels.matmulFloat(m, k, n, aData.buffer, bData.buffer, outBuffer)
        val outData = DenseFloatArrayTensorData<T>(Shape(m, n), outBuffer)
        @Suppress("UNCHECKED_CAST")
        return CpuTensor(outData as TensorData<T, V>, this, a.dtype)
    }

    private fun <T : DType, V> vectorFloatReduceAllSum(tensor: Tensor<T, V>): Tensor<T, V>? {
        if (!supportsFloatOps(tensor)) return null
        val data = tensor.data as? FloatArrayTensorData<T> ?: return null
        val buffer = data.buffer
        val n = buffer.size
        if (n == 0) return null
        // NOTE: For numerical reproducibility with Kotlin's FloatArray.sum(),
        // perform strict left-to-right scalar accumulation.
        var acc = 0.0f
        var idx = 0
        while (idx < n) {
            acc += buffer[idx]
            idx++
        }
        val outData = DenseFloatArrayTensorData<T>(Shape(), floatArrayOf(acc))
        @Suppress("UNCHECKED_CAST")
        return CpuTensor(outData as TensorData<T, V>, this, tensor.dtype)
    }
}
