package sk.ainet.exec.kernel.jni

import java.nio.ByteBuffer
import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.KernelKey
import sk.ainet.backend.api.kernel.LayoutClass
import sk.ainet.backend.api.kernel.OperandKey
import sk.ainet.backend.api.kernel.ReferenceMatmulKernel
import sk.ainet.backend.api.kernel.ViewKernel
import sk.ainet.lang.memory.BlockOrder
import sk.ainet.lang.memory.DirectBufferStorage
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.MappedBufferStorage
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * A [ViewKernel] over the JNI **direct-buffer row-major** matmuls (#1189): the weight is a
 * `BLOCKED_ROW_MAJOR` packed view whose storage is off-heap ([MappedBufferStorage] — mmap'd GGUF
 * pages — or [DirectBufferStorage]), read by the native kernel straight at its address. No heap
 * `ByteArray`, no relayout: this is what lets a model bigger than the ART cap decode on Android.
 *
 * Contrast with [sk.ainet.backend.api.kernel.PackedViewMatmulKernel], which serves the same
 * encodings from heap bytes in `BLOCKED_INPUT_MAJOR` (prepacked feed) order. Registering both
 * lets the dispatcher pick by what the weight actually is — canonical mapped weights match this
 * key exactly and are served with zero copies.
 */
@ExperimentalMemoryApi
public class JniBufferPackedMatmulKernel(
    encodingName: String,
    override val key: KernelKey,
    private val matmul: (
        input: FloatArray, inputOffset: Int,
        weight: ByteBuffer, weightByteOffset: Int,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    ) -> Unit,
) : ViewKernel {

    override val name: String = "jni-buffer-$encodingName"

    override fun run(inputs: List<TensorView>, out: TensorView) {
        require(inputs.size == 2) { "matmul takes two operands" }
        val a = inputs[0]
        val w = inputs[1]
        val rows = a.shape[0]
        val k = a.shape[1]
        val n = w.shape[0]
        require(w.shape[1] == k) { "inner dimensions disagree: [$rows, $k] × [$n, ${w.shape[1]}]" }
        check(w.layout.blockOrder == BlockOrder.ROW_MAJOR) {
            "$name reads canonical row-major weights; this one is ${w.layout.blockOrder}"
        }

        val buffer = when (val s = w.storage) {
            is MappedBufferStorage -> s.buffer()
            is DirectBufferStorage -> s.buffer()
            else -> return fallback(inputs, out)
        }
        val aHeap = a.storage as? Storage.Heap ?: return fallback(inputs, out)
        val oHeap = out.storage as? Storage.Heap ?: return fallback(inputs, out)
        val activation = aHeap.floats ?: return fallback(inputs, out)
        val output = oHeap.floats ?: return fallback(inputs, out)
        // The JNI kernel takes a contiguous activation row; a strided one would be mis-indexed.
        if (!a.isContiguous) return fallback(inputs, out)

        val weightOffset = (w.layout.offsetElements * w.layout.elementBytes).toInt()
        for (r in 0 until rows) {
            matmul(
                activation, aHeap.arrayOffset + (a.layout.offsetElements + r.toLong() * k).toInt(),
                buffer, weightOffset,
                k, n,
                output, oHeap.arrayOffset + (out.layout.offsetElements + r.toLong() * n).toInt(),
            )
        }
    }

    private fun fallback(inputs: List<TensorView>, out: TensorView) {
        ReferenceMatmulKernel(key).run(inputs, out)
    }

    public companion object {
        /** The key this kernel serves: dense contiguous FP32 activation × row-major packed weight. */
        public fun keyFor(encoding: TensorEncoding): KernelKey = KernelKey(
            op = "matmul",
            operands = listOf(
                OperandKey.contiguous(Format.dense(FP32)),
                OperandKey(Format(FP32, encoding), LayoutClass.BLOCKED_ROW_MAJOR),
            ),
        )
    }
}

/**
 * Registers the direct-buffer row-major kernels (#1189) into [KernelDispatch]. Call next to
 * `KernelPacks.install(JniKernelProvider)` on Android when weights are loaded with
 * `WeightResidency.MAPPED` — without it a mapped packed weight falls back to the decoding
 * reference kernel (correct, hours-scale slow).
 */
@ExperimentalMemoryApi
public object JniMappedKernelPack {
    public fun install() {
        if (!JniKernelProvider.isAvailable()) return
        KernelDispatch.register(
            JniBufferPackedMatmulKernel(
                TensorEncoding.Q4_K.name,
                JniBufferPackedMatmulKernel.keyFor(TensorEncoding.Q4_K),
                JniKernels::q4kMatmulRmDirect,
            ),
        )
        KernelDispatch.register(
            JniBufferPackedMatmulKernel(
                TensorEncoding.Q6_K.name,
                JniBufferPackedMatmulKernel.keyFor(TensorEncoding.Q6_K),
                JniKernels::q6kMatmulRmDirect,
            ),
        )
    }
}
