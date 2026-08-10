package sk.ainet.exec.kernel.jni

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import sk.ainet.exec.kernel.ScalarQ8_0MatmulKernel
import kotlin.random.Random

/**
 * On-device decode-throughput projection for SmolLM2-135M-Instruct (Q8_0) —
 * the acceptance measurement for #920 (field report: 1.0 tok/s scalar on
 * Android; usable ≥ ~3 tok/s).
 *
 * Autoregressive decode of a 135M model is dominated by the per-layer
 * projection mat-vecs plus the lm_head; attention/softmax/RoPE/sampling are
 * negligible at this size and short context. This benchmark runs the ACTUAL
 * JNI NEON kernels on the ACTUAL phone CPU at SmolLM2's real weight shapes,
 * sums one token's matmul wall-clock across all 30 layers + lm_head, and
 * reports projected tok/s for the JNI path and the scalar floor.
 *
 * It is a kernel-throughput projection, NOT an end-to-end generation (that
 * needs the full transformers stack — transformers#272). But Q8_0 mat-vec is
 * memory-bound, so timing at real shapes captures the real bottleneck.
 *
 * Results go to logcat under tag SKAINET_BENCH (dump after the run).
 */
@RunWith(AndroidJUnit4::class)
class SmolLm2DecodeBenchmark {

    private val tag = "SKAINET_BENCH"

    // SmolLM2-135M-Instruct config (HF): hidden 576, intermediate 1536,
    // 30 layers, 9 heads / 3 kv heads, head_dim 64, vocab 49152.
    private val hidden = 576
    private val inter = 1536
    private val layers = 30
    private val kvDim = 3 * 64      // 192
    private val vocab = 49152

    /** (inputDim, outputDim, perTokenCount) for each decode mat-vec. */
    private data class Shape(val inDim: Int, val outDim: Int, val perToken: Int, val name: String)

    private fun shapes(): List<Shape> = listOf(
        Shape(hidden, hidden, layers, "q_proj"),
        Shape(hidden, kvDim, layers, "k_proj"),
        Shape(hidden, kvDim, layers, "v_proj"),
        Shape(hidden, hidden, layers, "o_proj"),
        Shape(hidden, inter, layers, "gate_proj"),
        Shape(hidden, inter, layers, "up_proj"),
        Shape(inter, hidden, layers, "down_proj"),
        Shape(hidden, vocab, 1, "lm_head"),
    )

    /** Q8_0 packed weight: (inDim/32)*outDim blocks × 34 bytes, scale pinned to 1.0. */
    private fun q8Weight(inDim: Int, outDim: Int, seed: Int): ByteArray {
        val blocks = (inDim / 32) * outDim
        val bytes = ByteArray(blocks * 34)
        Random(seed).nextBytes(bytes)
        for (b in 0 until blocks) { bytes[b * 34] = 0x00; bytes[b * 34 + 1] = 0x3C }
        return bytes
    }

    private inline fun timeMedianNs(iters: Int, warmup: Int, body: () -> Unit): Long {
        repeat(warmup) { body() }
        val samples = LongArray(iters)
        for (i in 0 until iters) {
            val t0 = System.nanoTime()
            body()
            samples[i] = System.nanoTime() - t0
        }
        samples.sort()
        return samples[iters / 2]
    }

    @Test
    fun projected_decode_tokens_per_second() {
        assertTrue("JNI provider unavailable", JniKernelProvider.isAvailable())
        val variant = JniKernels.variant
        Log.i(tag, "=== SmolLM2-135M Q8_0 decode projection — device tier: $variant ===")

        // Pre-warm both kernels so ART's JIT has compiled them before the
        // first timed shape (otherwise the first scalar shape absorbs the
        // C2 compile and reads ~5x slow).
        run {
            val w = q8Weight(576, 576, 1)
            val i = FloatArray(576); val o = FloatArray(576)
            repeat(50) { JniKernels.q80Matmul(i, 0, w, 0, 576, 576, o, 0) }
            repeat(50) { ScalarQ8_0MatmulKernel.matmul(i, 0, w, 0, 576, 576, o, 0) }
        }

        var jniPerTokenNs = 0.0
        var scalarPerTokenNs = 0.0

        for (s in shapes()) {
            val w = q8Weight(s.inDim, s.outDim, seed = s.name.hashCode())
            val input = FloatArray(s.inDim) { Random(it).nextFloat() - 0.5f }
            val out = FloatArray(s.outDim)

            val jniNs = timeMedianNs(iters = 25, warmup = 5) {
                JniKernels.q80Matmul(input, 0, w, 0, s.inDim, s.outDim, out, 0)
            }
            // Scalar is far slower; fewer iters to keep the run bounded.
            val scalarNs = timeMedianNs(iters = 5, warmup = 2) {
                ScalarQ8_0MatmulKernel.matmul(input, 0, w, 0, s.inDim, s.outDim, out, 0)
            }

            jniPerTokenNs += jniNs.toDouble() * s.perToken
            scalarPerTokenNs += scalarNs.toDouble() * s.perToken
            Log.i(
                tag,
                "%-10s %5d->%-5d ×%-2d  jni=%7.1fµs  scalar=%8.1fµs  speedup=%.1fx".format(
                    s.name, s.inDim, s.outDim, s.perToken,
                    jniNs / 1000.0, scalarNs / 1000.0, scalarNs.toDouble() / jniNs,
                ),
            )
        }

        val jniTokPerSec = 1e9 / jniPerTokenNs
        val scalarTokPerSec = 1e9 / scalarPerTokenNs
        Log.i(tag, "--------------------------------------------------------------")
        Log.i(tag, "per-token matmul time:  jni=%.2f ms   scalar=%.2f ms".format(jniPerTokenNs / 1e6, scalarPerTokenNs / 1e6))
        Log.i(tag, "PROJECTED DECODE:       jni=%.2f tok/s   scalar=%.2f tok/s   (%.1fx)".format(jniTokPerSec, scalarTokPerSec, jniTokPerSec / scalarTokPerSec))
        Log.i(tag, "usability gate (3 tok/s): jni %s".format(if (jniTokPerSec >= 3.0) "PASS ✅" else "FAIL ❌"))

        assertTrue("projected tok/s must be positive", jniTokPerSec > 0)
    }
}
