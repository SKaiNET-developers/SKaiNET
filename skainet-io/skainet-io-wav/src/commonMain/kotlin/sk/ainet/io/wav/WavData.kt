package sk.ainet.io.wav

import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.Tensor

public data class WavData(
    val samples: FloatArray,
    val sampleRate: Int
)

/**
 * Reads a 16-bit PCM WAV file (mono or stereo). Stereo data is averaged to mono.
 */
public expect fun readWav(path: String): WavData

/**
 * Utility to lift wav samples into a tensor on the provided execution context.
 */
public fun WavData.asTensor(factory: (FloatArray) -> Tensor<FP32, Float>): Tensor<FP32, Float> =
    factory(samples)
