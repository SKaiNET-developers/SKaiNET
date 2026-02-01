package sk.ainet.lang.audio

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.FP32
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Configuration for MFCC extraction.
 */
public data class MfccConfig(
    val sampleRate: Int,
    val frameSize: Int = 400,
    val hopSize: Int = 160,
    val fftSize: Int = 512,
    val melBands: Int = 40,
    val coeffs: Int = 13,
    val preEmphasis: Float = 0.97f,
    val includeEnergy: Boolean = false,
    val deltas: Boolean = false,
    val deltaWindow: Int = 2
) {
    init {
        require(frameSize > 0) { "frameSize must be positive" }
        require(hopSize > 0) { "hopSize must be positive" }
        require(fftSize >= frameSize) { "fftSize must be >= frameSize" }
        require(melBands > 0) { "melBands must be positive" }
        require(coeffs > 0) { "coeffs must be positive" }
        require(sampleRate > 0) { "sampleRate must be positive" }
        require(deltaWindow >= 1) { "deltaWindow must be at least 1" }
    }

    internal val fftBins: Int get() = (fftSize / 2) + 1
    internal val staticCoeffCount: Int get() = coeffs + if (includeEnergy) 1 else 0
    internal val totalCoeffCount: Int
        get() = staticCoeffCount * if (deltas) 3 else 1
}

/**
 * Precomputed windows and transforms used by MFCC extraction.
 */
public class AudioPlan internal constructor(
    public val config: MfccConfig,
    public val hannWindow: FloatArray,
    public val melFilterbank: FloatArray,
    public val dctMatrix: FloatArray
) {
    public companion object {
        public fun build(config: MfccConfig): AudioPlan {
            val hann = hann(config.frameSize)
            val mel = melFilterbank(config)
            val dct = dctMatrix(config.melBands)
            return AudioPlan(config, hann, mel, dct)
        }

        private fun hann(size: Int): FloatArray =
            FloatArray(size) { i ->
                (0.5f - 0.5f * cos(((2.0 * PI * i) / (size - 1)).toFloat()))
            }

        private fun melFilterbank(config: MfccConfig): FloatArray {
            val fftBins = config.fftBins
            val melPoints = FloatArray(config.melBands + 2) { idx ->
                val melMin = hzToMel(0.0f)
                val melMax = hzToMel((config.sampleRate / 2.0f))
                melMin + (melMax - melMin) * idx / (config.melBands + 1)
            }
            val hzPoints = FloatArray(melPoints.size) { melToHz(melPoints[it]) }
            val binPoints = IntArray(hzPoints.size) { idx ->
                floor(((config.fftSize + 1) * hzPoints[idx]) / config.sampleRate).toInt()
            }

            val filters = FloatArray(config.melBands * fftBins)
            for (m in 1..config.melBands) {
                val left = binPoints[m - 1]
                val center = binPoints[m]
                val right = binPoints[m + 1]
                val filterBase = (m - 1) * fftBins

                if (center > left) {
                    for (k in left until center) {
                        val weight = (k - left).toFloat() / (center - left)
                        if (k in 0 until fftBins) {
                            filters[filterBase + k] = weight
                        }
                    }
                }
                if (right > center) {
                    for (k in center until right) {
                        val weight = (right - k).toFloat() / (right - center)
                        if (k in 0 until fftBins) {
                            filters[filterBase + k] = max(filters[filterBase + k], weight)
                        }
                    }
                }
            }
            return filters
        }

        private fun dctMatrix(size: Int): FloatArray {
            val scale0 = sqrt(1.0f / size)
            val scale = sqrt(2.0f / size)
            val mat = FloatArray(size * size)
            for (k in 0 until size) {
                val scaleK = if (k == 0) scale0 else scale
                for (n in 0 until size) {
                    mat[k * size + n] =
                        scaleK * cos(((PI / size) * (n + 0.5) * k).toFloat())
                }
            }
            return mat
        }

        private fun hzToMel(hz: Float): Float = 2595f * ln(1f + hz / 700f)
        private fun melToHz(mel: Float): Float = 700f * ((kotlin.math.exp(mel / 2595f) - 1f))
    }
}

/**
 * Backend contract so specialized FFT/DCT implementations can be swapped in.
 */
public interface AudioBackend {
    public fun mfcc(
        context: ExecutionContext,
        signal: Tensor<FP32, Float>,
        config: MfccConfig,
        plan: AudioPlan = AudioPlan.build(config)
    ): Tensor<FP32, Float>
}

/**
 * Reference backend implemented with dense host math. Optimized backends can
 * implement [AudioBackend] to dispatch to platform FFT/DCT kernels.
 */
public object ReferenceAudioBackend : AudioBackend {
    private const val EPSILON: Float = 1e-10f

    override fun mfcc(
        context: ExecutionContext,
        signal: Tensor<FP32, Float>,
        config: MfccConfig,
        plan: AudioPlan
    ): Tensor<FP32, Float> {
        val (batch, samples) = when (signal.shape.rank) {
            1 -> 1 to signal.shape[0]
            2 -> signal.shape[0] to signal.shape[1]
            else -> error("MFCC expects shape [batch, samples] or [samples], got ${signal.shape}")
        }
        require(samples >= config.frameSize) {
            "Not enough samples (${samples}) for frame size ${config.frameSize}"
        }
        val frames = 1 + (samples - config.frameSize) / config.hopSize
        val fftBins = config.fftBins
        val staticCount = config.staticCoeffCount
        val coeffCount = config.totalCoeffCount
        val output = FloatArray(batch * frames * coeffCount)

        for (b in 0 until batch) {
            val mono = preEmphasize(extractBatch(signal, b, samples), config.preEmphasis)
            val static = FloatArray(frames * staticCount)
            for (frameIdx in 0 until frames) {
                val frameStart = frameIdx * config.hopSize
                val windowed = FloatArray(config.fftSize)
                for (i in 0 until config.frameSize) {
                    windowed[i] = mono[frameStart + i] * plan.hannWindow[i]
                }
                val powerSpectrum = powerSpectrum(windowed, config.fftSize, fftBins)
                val melEnergies = melEnergies(powerSpectrum, plan.melFilterbank, plan.config.melBands, fftBins)
                val logEnergy = ln(max(powerSpectrum.sum(), EPSILON))
                val logMel = FloatArray(plan.config.melBands) { idx ->
                    ln(max(melEnergies[idx], EPSILON))
                }
                val dct = FloatArray(plan.config.melBands)
                for (k in 0 until plan.config.melBands) {
                    var acc = 0f
                    val rowOffset = k * plan.config.melBands
                    for (m in 0 until plan.config.melBands) {
                        acc += plan.dctMatrix[rowOffset + m] * logMel[m]
                    }
                    dct[k] = acc
                }

                val frameOffset = frameIdx * staticCount
                var cursor = frameOffset
                if (config.includeEnergy) {
                    static[cursor++] = logEnergy
                }
                for (c in 0 until config.coeffs) {
                    static[cursor++] = dct[c]
                }
            }

            val finalBlock = if (config.deltas) {
                val delta = deltas(static, frames, staticCount, config.deltaWindow)
                val deltaDelta = deltas(delta, frames, staticCount, config.deltaWindow)
                val concat = FloatArray(frames * coeffCount)
                copyBlock(static, staticCount, frames, 0, concat)
                copyBlock(delta, staticCount, frames, staticCount, concat)
                copyBlock(deltaDelta, staticCount, frames, staticCount * 2, concat)
                concat
            } else {
                static
            }

            val batchOffset = b * frames * coeffCount
            finalBlock.copyInto(
                destination = output,
                destinationOffset = batchOffset,
                endIndex = batchOffset + finalBlock.size
            )
        }

        return context.fromFloatArray(
            Shape(batch, frames, coeffCount),
            FP32::class,
            output
        )
    }

    private fun extractBatch(tensor: Tensor<FP32, Float>, batchIndex: Int, samples: Int): FloatArray {
        val data = tensor.data
        return if (tensor.shape.rank == 1) {
            when (data) {
                is FloatArrayTensorData<*> -> data.buffer.copyOf()
                else -> FloatArray(samples) { idx -> data.get(idx) }
            }
        } else {
            when (data) {
                is FloatArrayTensorData<*> -> {
                    val offset = batchIndex * samples
                    data.buffer.copyOfRange(offset, offset + samples)
                }
                else -> FloatArray(samples) { idx -> data.get(batchIndex, idx) }
            }
        }
    }

    private fun preEmphasize(samples: FloatArray, factor: Float): FloatArray {
        if (factor == 0f) return samples.copyOf()
        val out = FloatArray(samples.size)
        if (samples.isNotEmpty()) {
            out[0] = samples[0]
            for (i in 1 until samples.size) {
                out[i] = samples[i] - factor * samples[i - 1]
            }
        }
        return out
    }

    private fun powerSpectrum(frame: FloatArray, fftSize: Int, fftBins: Int): FloatArray {
        val twoPi = (2.0 * PI).toFloat()
        val power = FloatArray(fftBins)
        for (k in 0 until fftBins) {
            var real = 0f
            var imag = 0f
            for (n in 0 until fftSize) {
                val angle = twoPi * k * n / fftSize
                val sample = frame[n]
                real += sample * cos(angle)
                imag -= sample * sin(angle)
            }
            power[k] = real * real + imag * imag
        }
        return power
    }

    private fun melEnergies(
        power: FloatArray,
        filterbank: FloatArray,
        melBands: Int,
        fftBins: Int
    ): FloatArray {
        val mel = FloatArray(melBands)
        for (m in 0 until melBands) {
            var acc = 0f
            val base = m * fftBins
            for (k in 0 until fftBins) {
                acc += power[k] * filterbank[base + k]
            }
            mel[m] = acc
        }
        return mel
    }

    private fun deltas(source: FloatArray, frames: Int, coeffs: Int, window: Int): FloatArray {
        val out = FloatArray(source.size)
        val denom = (1..window).sumOf { it * it } * 2
        for (frame in 0 until frames) {
            for (c in 0 until coeffs) {
                var num = 0f
                for (n in 1..window) {
                    val prev = source[indexOf(frame - n, coeffs, c, frames)]
                    val next = source[indexOf(frame + n, coeffs, c, frames)]
                    num += n * (next - prev)
                }
                out[indexOf(frame, coeffs, c, frames)] = num / denom
            }
        }
        return out
    }

    private fun copyBlock(
        block: FloatArray,
        coeffs: Int,
        frames: Int,
        coeffOffset: Int,
        out: FloatArray
    ) {
        for (f in 0 until frames) {
            val src = f * coeffs
            val dst = f * (coeffs * 3) + coeffOffset
            block.copyInto(out, destinationOffset = dst, startIndex = src, endIndex = src + coeffs)
        }
    }

    private fun FloatArray.sum(): Float {
        var acc = 0f
        for (v in this) acc += v
        return acc
    }

    private fun indexOf(frame: Int, coeffs: Int, coeff: Int, frames: Int): Int {
        val clampedFrame = when {
            frame < 0 -> 0
            frame >= frames -> frames - 1
            else -> frame
        }
        return clampedFrame * coeffs + coeff
    }
}

/**
 * Convenience entry point that defaults to the reference backend.
 */
public fun mfcc(
    context: ExecutionContext,
    signal: Tensor<FP32, Float>,
    config: MfccConfig,
    plan: AudioPlan = AudioPlan.build(config),
    backend: AudioBackend = ReferenceAudioBackend
): Tensor<FP32, Float> = backend.mfcc(context, signal, config, plan)
