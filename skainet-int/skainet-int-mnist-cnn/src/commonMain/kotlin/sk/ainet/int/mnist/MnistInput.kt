package sk.ainet.int.mnist

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.dsl.tensor
import sk.ainet.lang.types.FP32

/**
 * Minimal input helpers for shaping/normalizing 1x28x28 grayscale inputs (BCHW order).
 * These functions are part of the public API per PRD (mnist-int.md, section 2.3).
 */
object MnistInputHelpers {
    const val WIDTH: Int = 28
    const val HEIGHT: Int = 28
    const val CHANNELS: Int = 1

    /**
     * Creates a single-sample FP32 tensor with shape [1, 1, 28, 28] from a 28x28 grayscale image
     * represented as a ByteArray of length 784. If [normalize] is true (default), values are mapped
     * from [0,255] to [0f,1f] by dividing by 255f.
     */
    fun toMnistTensor(exec: ExecutionContext, bytes: ByteArray, normalize: Boolean = true): Tensor<FP32, Float> {
        require(bytes.size == WIDTH * HEIGHT) { "Expected ${WIDTH * HEIGHT} bytes for 28x28 image, got ${bytes.size}" }
        val floats = FloatArray(bytes.size) { idx ->
            val v = (bytes[idx].toInt() and 0xFF).toFloat()
            if (normalize) v / 255f else v
        }
        return toMnistTensor(exec, floats, normalized = true)
    }

    /**
     * Creates a single-sample FP32 tensor with shape [1, 1, 28, 28] from a 28x28 grayscale image
     * represented as a FloatArray of length 784. If [normalized] is false (default true), values are
     * assumed to be in [0,255] and will be divided by 255f. Values are clamped to [0f,1f].
     */
    fun toMnistTensor(exec: ExecutionContext, floats: FloatArray, normalized: Boolean = true): Tensor<FP32, Float> {
        require(floats.size == WIDTH * HEIGHT) { "Expected ${WIDTH * HEIGHT} floats for 28x28 image, got ${floats.size}" }
        val prepared = if (normalized) floats.copyOf() else FloatArray(floats.size) { i -> floats[i] / 255f }
        // Clamp to [0,1]
        for (i in prepared.indices) {
            val v = prepared[i]
            prepared[i] = when {
                v < 0f -> 0f
                v > 1f -> 1f
                else -> v
            }
        }
        // BCHW: [1, 1, H, W]
        return tensor(exec, FP32::class) {
            tensor {
                shape(1, CHANNELS, HEIGHT, WIDTH) {
                    fromArray(prepared)
                }
            }
        }
    }

    /**
     * Stacks multiple single-sample [1,1,28,28] tensors into a batch tensor with shape [N,1,28,28].
     * Accepts any tensor implementation; tries to fast-path when buffer is accessible.
     */
    fun stackBatch(exec: ExecutionContext, vararg samples: Tensor<FP32, Float>): Tensor<FP32, Float> {
        require(samples.isNotEmpty()) { "At least one sample is required" }
        val n = samples.size
        samples.forEachIndexed { idx, t ->
            val s = t.shape
            require(s.rank == 4 && s[0] == 1 && s[1] == CHANNELS && s[2] == HEIGHT && s[3] == WIDTH) {
                "Sample at index $idx has invalid shape $s; expected [1,1,28,28]"
            }
        }
        val perSample = CHANNELS * HEIGHT * WIDTH
        val buf = FloatArray(n * perSample)
        var offset = 0
        for (t in samples) {
            val data = t.data
            if (data is FloatArrayTensorData<*>) {
                val src = data.buffer
                // data is [1,1,28,28] contiguous; copy all
                // ensure volume matches expected
                require(src.size == perSample) { "Unexpected buffer size ${src.size} for sample (expected $perSample)" }
                src.copyInto(buf, destinationOffset = offset)
                offset += perSample
            } else {
                // Fallback: index via accessor
                for (c in 0 until CHANNELS) {
                    for (h in 0 until HEIGHT) {
                        for (w in 0 until WIDTH) {
                            buf[offset++] = data[0, c, h, w]
                        }
                    }
                }
            }
        }
        return tensor(exec, FP32::class) {
            tensor {
                shape(n, CHANNELS, HEIGHT, WIDTH) {
                    fromArray(buf)
                }
            }
        }
    }
}
