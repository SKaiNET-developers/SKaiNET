package sk.ainet.docs.samples

// tag::imports[]
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.data
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.tensor.matmul
import sk.ainet.lang.tensor.plus
import sk.ainet.lang.tensor.relu
import sk.ainet.lang.tensor.reshape
import sk.ainet.lang.tensor.t
import sk.ainet.lang.types.FP32
// end::imports[]

/**
 * Tensor construction and the everyday eager operations, all real and CI-run.
 * Regions here are included by `how-to/tensor-ops.adoc` and the usage-examples page.
 */
object TensorBasics {

    /** Build a single 2x2 tensor with the `data { }` DSL — the idiomatic form. */
    // tag::create-one[]
    fun oneTensor(ctx: DirectCpuExecutionContext): Tensor<FP32, Float> =
        data<FP32, Float>(ctx) {
            tensor {
                shape(2, 2) { from(1f, 2f, 3f, 4f) }
            }
        }
    // end::create-one[]

    /** The initialization strategies available inside `shape(...) { ... }`. */
    // tag::init[]
    fun initStrategies(ctx: DirectCpuExecutionContext): List<Tensor<FP32, Float>> {
        lateinit var zeros: Tensor<FP32, Float>
        lateinit var ones: Tensor<FP32, Float>
        lateinit var filled: Tensor<FP32, Float>
        lateinit var gaussian: Tensor<FP32, Float>
        lateinit var ramp: Tensor<FP32, Float>
        data(ctx) {
            zeros = tensor { shape(2, 3) { zeros() } }
            ones = tensor { shape(2, 3) { ones() } }
            filled = tensor { shape(2, 3) { full(0.5f) } }
            gaussian = tensor { shape(2, 3) { randn(mean = 0f, std = 0.02f) } }
            ramp = tensor { shape(2, 3) { init { idx -> (idx[0] + idx[1]).toFloat() } } }
        }
        return listOf(zeros, ones, filled, gaussian, ramp)
    }
    // end::init[]

    /**
     * Everyday eager ops: matrix multiply, transpose, reshape, ReLU. Each op is an
     * extension on `Tensor` — no execution context to thread through, the tensor
     * carries its own backend ops.
     */
    // tag::ops[]
    fun ops(ctx: DirectCpuExecutionContext): Tensor<FP32, Float> {
        lateinit var a: Tensor<FP32, Float>
        lateinit var b: Tensor<FP32, Float>
        data(ctx) {
            a = tensor { shape(2, 3) { from(1f, 2f, 3f, 4f, 5f, 6f) } }
            b = tensor { shape(3, 2) { from(1f, 0f, 0f, 1f, 1f, 1f) } }
        }
        val product = a.matmul(b)        // [2,3] x [3,2] -> [2,2]
        val transposed = product.t()     // [2,2] -> [2,2]
        val flat = transposed.reshape(Shape(4))
        return flat.relu()
    }
    // end::ops[]

    /**
     * Broadcasting: a per-column bias of shape [1,3] is added across both rows of a
     * [2,3] matrix. A scalar broadcasts to every element.
     */
    // tag::broadcast[]
    fun broadcast(ctx: DirectCpuExecutionContext): Tensor<FP32, Float> {
        lateinit var matrix: Tensor<FP32, Float>
        lateinit var bias: Tensor<FP32, Float>
        data(ctx) {
            matrix = tensor { shape(2, 3) { from(1f, 2f, 3f, 4f, 5f, 6f) } }
            bias = tensor { shape(1, 3) { from(10f, 20f, 30f) } }
        }
        val biased = matrix + bias       // [2,3] + [1,3] -> [2,3]
        return biased + 100f             // scalar broadcasts to every element
    }
    // end::broadcast[]
}
