package sk.ainet.exec.tensor.ops

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

/**
 * Number of CPU cores available for kernel-level parallelism.
 * JVM-only for now; promote to expect/actual in commonMain when native/JS
 * backends gain SIMD kernels too.
 */
internal val defaultParallelism: Int = Runtime.getRuntime().availableProcessors()

/**
 * Threshold below which a matmul stays single-threaded — coroutine launch
 * overhead dominates for tiny outputDim. Tuned empirically: chunks below
 * this size are not worth dispatching.
 */
internal const val PARALLEL_MATMUL_MIN_OUTPUT: Int = 256

/**
 * Run [block] over disjoint chunks of [outputDim] in parallel.
 * Below [PARALLEL_MATMUL_MIN_OUTPUT] runs sequentially in the calling thread.
 *
 * Each task receives the half-open range `[start, end)` it owns.
 * Use [Dispatchers.Default] which is sized to CPU count on JVM.
 */
internal inline fun parallelChunks(
    outputDim: Int,
    crossinline block: (start: Int, end: Int) -> Unit
) {
    if (outputDim < PARALLEL_MATMUL_MIN_OUTPUT) {
        block(0, outputDim)
        return
    }
    val chunks = defaultParallelism
    val chunkSize = (outputDim + chunks - 1) / chunks
    runBlocking(Dispatchers.Default) {
        coroutineScope {
            var start = 0
            while (start < outputDim) {
                val end = minOf(start + chunkSize, outputDim)
                val s = start
                val e = end
                launch { block(s, e) }
                start = end
            }
        }
    }
}
