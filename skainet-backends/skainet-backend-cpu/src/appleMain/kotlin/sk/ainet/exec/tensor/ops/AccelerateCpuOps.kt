@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package sk.ainet.exec.tensor.ops

import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import platform.Accelerate.CblasNoTrans
import platform.Accelerate.CblasRowMajor
import platform.Accelerate.cblas_sgemm
import platform.Accelerate.vDSP_vadd
import platform.Accelerate.vDSP_vsub
import platform.Accelerate.vDSP_vmul
import platform.Accelerate.vDSP_vdiv
import platform.Accelerate.vDSP_sve
import platform.Accelerate.vDSP_meanv
import platform.Accelerate.vDSP_mtrans
import platform.Accelerate.vDSP_vthres
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.TensorDataFactory
import sk.ainet.lang.ops.Backend
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

/**
 * CPU operations accelerated by Apple's Accelerate framework.
 * Overrides hot-path operations (matmul, elementwise, reductions) with
 * hardware-optimized routines that leverage ARM NEON and AMX.
 *
 * Falls through to [DefaultCpuOpsBase] for non-FP32, non-contiguous,
 * or complex broadcasting cases.
 */
@Backend(id = "apple", displayName = "Apple Accelerate")
public class AccelerateCpuOps(
    dataFactory: TensorDataFactory,
    schedule: sk.ainet.context.schedule.Schedule,
) : DefaultCpuOpsBase(dataFactory, schedule) {
    public constructor(dataFactory: TensorDataFactory) : this(dataFactory, sk.ainet.context.schedule.Schedule.Sequential)


    // ── matmul ──────────────────────────────────────────────────────────

    override fun <T : DType, V> matmul(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        if (a.rank == 2 && b.rank == 2
            && a.dtype == FP32::class
            && a.data is FloatArrayTensorData<*>
            && b.data is FloatArrayTensorData<*>
        ) {
            val aBuf = (a.data as FloatArrayTensorData<*>).buffer
            val bBuf = (b.data as FloatArrayTensorData<*>).buffer
            val m = a.shape[0]
            val k = a.shape[1]
            val n = b.shape[1]
            require(k == b.shape[0]) { "matmul shape mismatch: ${a.shape} vs ${b.shape}" }

            val out = FloatArray(m * n)
            // cblas_sgemm: C = alpha * A * B + beta * C
            aBuf.usePinned { aPin ->
                bBuf.usePinned { bPin ->
                    out.usePinned { cPin ->
                        cblas_sgemm(
                            CblasRowMajor,
                            CblasNoTrans, CblasNoTrans,
                            m, n, k,
                            1.0f,                       // alpha
                            aPin.addressOf(0), k,       // A, lda
                            bPin.addressOf(0), n,       // B, ldb
                            0.0f,                       // beta
                            cPin.addressOf(0), n,       // C, ldc
                        )
                    }
                }
            }

            @Suppress("UNCHECKED_CAST")
            val outData = dataFactory.fromFloatArray<T, Float>(Shape(m, n), a.dtype, out)
                    as sk.ainet.lang.tensor.data.TensorData<T, V>
            return newTensor(outData, a.dtype, a, b)
        }

        return super.matmul(a, b)
    }

    // ── elementwise binary ops ──────────────────────────────────────────

    override fun <T : DType, V> add(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val result = tryVdspBinary(a, b, ::vdspAdd)
        return result ?: super.add(a, b)
    }

    override fun <T : DType, V> subtract(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val result = tryVdspBinary(a, b, ::vdspSub)
        return result ?: super.subtract(a, b)
    }

    override fun <T : DType, V> multiply(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val result = tryVdspBinary(a, b, ::vdspMul)
        return result ?: super.multiply(a, b)
    }

    override fun <T : DType, V> divide(a: Tensor<T, V>, b: Tensor<T, V>): Tensor<T, V> {
        val result = tryVdspBinary(a, b, ::vdspDiv)
        return result ?: super.divide(a, b)
    }

    // ── reductions ──────────────────────────────────────────────────────

    override fun <T : DType, V> sum(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        if (dim == null
            && tensor.dtype == FP32::class
            && tensor.data is FloatArrayTensorData<*>
        ) {
            val buf = (tensor.data as FloatArrayTensorData<*>).buffer
            val n = buf.size
            if (n > 0) {
                val result = FloatArray(1)
                buf.usePinned { pin ->
                    result.usePinned { rPin ->
                        vDSP_sve(pin.addressOf(0), 1, rPin.addressOf(0), n.toULong())
                    }
                }
                @Suppress("UNCHECKED_CAST")
                val outData = dataFactory.fromFloatArray<T, Float>(Shape(), tensor.dtype, floatArrayOf(result[0]))
                        as sk.ainet.lang.tensor.data.TensorData<T, V>
                return newTensor(outData, tensor.dtype, tensor)
            }
        }
        return super.sum(tensor, dim)
    }

    override fun <T : DType, V> mean(tensor: Tensor<T, V>, dim: Int?): Tensor<T, V> {
        if (dim == null
            && tensor.dtype == FP32::class
            && tensor.data is FloatArrayTensorData<*>
        ) {
            val buf = (tensor.data as FloatArrayTensorData<*>).buffer
            val n = buf.size
            if (n > 0) {
                val result = FloatArray(1)
                buf.usePinned { pin ->
                    result.usePinned { rPin ->
                        vDSP_meanv(pin.addressOf(0), 1, rPin.addressOf(0), n.toULong())
                    }
                }
                @Suppress("UNCHECKED_CAST")
                val outData = dataFactory.fromFloatArray<T, Float>(Shape(), tensor.dtype, floatArrayOf(result[0]))
                        as sk.ainet.lang.tensor.data.TensorData<T, V>
                return newTensor(outData, tensor.dtype, tensor)
            }
        }
        return super.mean(tensor, dim)
    }

    // ── activations ─────────────────────────────────────────────────────

    override fun <T : DType, V> relu(tensor: Tensor<T, V>): Tensor<T, V> {
        if (tensor.dtype == FP32::class && tensor.data is FloatArrayTensorData<*>) {
            val buf = (tensor.data as FloatArrayTensorData<*>).buffer
            val n = buf.size
            val out = FloatArray(n)
            buf.usePinned { pin ->
                out.usePinned { oPin ->
                    val threshold = FloatArray(1) { 0.0f }
                    threshold.usePinned { tPin ->
                        vDSP_vthres(pin.addressOf(0), 1, tPin.addressOf(0), oPin.addressOf(0), 1, n.toULong())
                    }
                }
            }
            @Suppress("UNCHECKED_CAST")
            val outData = dataFactory.fromFloatArray<T, Float>(tensor.shape, tensor.dtype, out)
                    as sk.ainet.lang.tensor.data.TensorData<T, V>
            return newTensor(outData, tensor.dtype, tensor)
        }
        return super.relu(tensor)
    }

    override fun <T : DType, V> silu(tensor: Tensor<T, V>): Tensor<T, V> {
        if (tensor.dtype == FP32::class && tensor.data is FloatArrayTensorData<*>) {
            val buf = (tensor.data as FloatArrayTensorData<*>).buffer
            val n = buf.size
            val out = FloatArray(n)
            for (i in 0 until n) {
                val x = buf[i]
                out[i] = x / (1.0f + kotlin.math.exp(-x))
            }
            @Suppress("UNCHECKED_CAST")
            val outData = dataFactory.fromFloatArray<T, Float>(tensor.shape, tensor.dtype, out)
                    as sk.ainet.lang.tensor.data.TensorData<T, V>
            return newTensor(outData, tensor.dtype, tensor)
        }
        return super.silu(tensor)
    }

    // ── transpose ───────────────────────────────────────────────────────

    override fun <T : DType, V> transpose(tensor: Tensor<T, V>): Tensor<T, V> {
        if (tensor.rank == 2
            && tensor.dtype == FP32::class
            && tensor.data is FloatArrayTensorData<*>
        ) {
            val buf = (tensor.data as FloatArrayTensorData<*>).buffer
            val rows = tensor.shape[0]
            val cols = tensor.shape[1]
            val out = FloatArray(rows * cols)
            buf.usePinned { pin ->
                out.usePinned { oPin ->
                    vDSP_mtrans(
                        pin.addressOf(0), 1,
                        oPin.addressOf(0), 1,
                        cols.toULong(), rows.toULong(),
                    )
                }
            }
            @Suppress("UNCHECKED_CAST")
            val outData = dataFactory.fromFloatArray<T, Float>(Shape(cols, rows), tensor.dtype, out)
                    as sk.ainet.lang.tensor.data.TensorData<T, V>
            return newTensor(outData, tensor.dtype, tensor)
        }
        return super.transpose(tensor)
    }

    // ── vDSP binary helpers ─────────────────────────────────────────────

    /**
     * Attempt to dispatch a binary elementwise op to vDSP.
     * Returns null if the tensors are not eligible (non-FP32, non-contiguous,
     * complex broadcasting).
     */
    private fun <T : DType, V> tryVdspBinary(
        a: Tensor<T, V>,
        b: Tensor<T, V>,
        op: (FloatArray, FloatArray, FloatArray, Int) -> Unit,
    ): Tensor<T, V>? {
        if (a.dtype != FP32::class) return null
        if (a.data !is FloatArrayTensorData<*> || b.data !is FloatArrayTensorData<*>) return null

        val aBuf = (a.data as FloatArrayTensorData<*>).buffer
        val bBuf = (b.data as FloatArrayTensorData<*>).buffer

        // Same shape: straightforward vectorized op
        if (a.shape == b.shape) {
            val n = aBuf.size
            val out = FloatArray(n)
            op(aBuf, bBuf, out, n)
            @Suppress("UNCHECKED_CAST")
            val outData = dataFactory.fromFloatArray<T, Float>(a.shape, a.dtype, out)
                    as sk.ainet.lang.tensor.data.TensorData<T, V>
            return newTensor(outData, a.dtype, a, b)
        }

        // Scalar broadcast: b is a single element
        if (bBuf.size == 1) {
            val n = aBuf.size
            val expanded = FloatArray(n) { bBuf[0] }
            val out = FloatArray(n)
            op(aBuf, expanded, out, n)
            @Suppress("UNCHECKED_CAST")
            val outData = dataFactory.fromFloatArray<T, Float>(a.shape, a.dtype, out)
                    as sk.ainet.lang.tensor.data.TensorData<T, V>
            return newTensor(outData, a.dtype, a, b)
        }

        // Scalar broadcast: a is a single element
        if (aBuf.size == 1) {
            val n = bBuf.size
            val expanded = FloatArray(n) { aBuf[0] }
            val out = FloatArray(n)
            op(expanded, bBuf, out, n)
            @Suppress("UNCHECKED_CAST")
            val outData = dataFactory.fromFloatArray<T, Float>(b.shape, a.dtype, out)
                    as sk.ainet.lang.tensor.data.TensorData<T, V>
            return newTensor(outData, a.dtype, a, b)
        }

        // Last-dim broadcast: b has shape [1, ..., 1, N] matching a's last dim
        // Common for bias add: [batch, features] + [features]
        if (b.rank <= a.rank) {
            val bDims = b.shape.dimensions
            val aDims = a.shape.dimensions
            val offset = aDims.size - bDims.size
            var isBiasBroadcast = true
            for (i in bDims.indices) {
                if (i < bDims.size - 1 && bDims[i] != 1) { isBiasBroadcast = false; break }
                if (i == bDims.size - 1 && bDims[i] != aDims[offset + i]) { isBiasBroadcast = false; break }
            }
            if (isBiasBroadcast && bDims.last() > 1) {
                val lastDim = bDims.last()
                val batches = aBuf.size / lastDim
                val out = FloatArray(aBuf.size)
                for (batch in 0 until batches) {
                    val aSlice = FloatArray(lastDim)
                    aBuf.copyInto(aSlice, 0, batch * lastDim, (batch + 1) * lastDim)
                    val oSlice = FloatArray(lastDim)
                    op(aSlice, bBuf, oSlice, lastDim)
                    oSlice.copyInto(out, batch * lastDim)
                }
                @Suppress("UNCHECKED_CAST")
                val outData = dataFactory.fromFloatArray<T, Float>(a.shape, a.dtype, out)
                        as sk.ainet.lang.tensor.data.TensorData<T, V>
                return newTensor(outData, a.dtype, a, b)
            }
        }

        return null // fall through to scalar
    }

    private fun vdspAdd(a: FloatArray, b: FloatArray, out: FloatArray, n: Int) {
        a.usePinned { aPin ->
            b.usePinned { bPin ->
                out.usePinned { oPin ->
                    vDSP_vadd(aPin.addressOf(0), 1, bPin.addressOf(0), 1, oPin.addressOf(0), 1, n.toULong())
                }
            }
        }
    }

    private fun vdspSub(a: FloatArray, b: FloatArray, out: FloatArray, n: Int) {
        // vDSP_vsub computes out = B - A (reversed!), so swap args
        a.usePinned { aPin ->
            b.usePinned { bPin ->
                out.usePinned { oPin ->
                    vDSP_vsub(bPin.addressOf(0), 1, aPin.addressOf(0), 1, oPin.addressOf(0), 1, n.toULong())
                }
            }
        }
    }

    private fun vdspMul(a: FloatArray, b: FloatArray, out: FloatArray, n: Int) {
        a.usePinned { aPin ->
            b.usePinned { bPin ->
                out.usePinned { oPin ->
                    vDSP_vmul(aPin.addressOf(0), 1, bPin.addressOf(0), 1, oPin.addressOf(0), 1, n.toULong())
                }
            }
        }
    }

    private fun vdspDiv(a: FloatArray, b: FloatArray, out: FloatArray, n: Int) {
        // vDSP_vdiv computes out = B / A (reversed!), so swap args
        a.usePinned { aPin ->
            b.usePinned { bPin ->
                out.usePinned { oPin ->
                    vDSP_vdiv(bPin.addressOf(0), 1, aPin.addressOf(0), 1, oPin.addressOf(0), 1, n.toULong())
                }
            }
        }
    }
}
