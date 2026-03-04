package sk.ainet.compile.hlo.generate

import sk.ainet.apps.kwhisper.WhisperModelMetadata
import sk.ainet.apps.kwhisper.dsl.WhisperEncoderHloModel
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.model.Model
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.model.compute.Rgb2GrayScale
import sk.ainet.lang.model.compute.Rgb2GrayScaleMatMul

/**
 * Descriptor for a registered model that can be compiled to StableHLO.
 */
internal data class ModelDescriptor(
    val name: String,
    val description: String,
    val functionName: String,
    val createModelAndInput: (ctx: ExecutionContext, height: Int, width: Int, batch: Int) -> ModelAndInput<*, *>
)

/**
 * Holds a model instance and a sample input tensor.
 */
internal data class ModelAndInput<D : DType, V>(
    val model: Model<D, V, Tensor<D, V>, Tensor<D, V>>,
    val sampleInput: Tensor<D, V>
)

/**
 * Registry of CLI-friendly model names to their descriptors.
 */
internal object ModelRegistry {

    private val models: Map<String, ModelDescriptor> = buildMap {
        put("rgb2grayscale", ModelDescriptor(
            name = "rgb2grayscale",
            description = "RGB to Grayscale via 1x1 Conv2D (FP32)",
            functionName = "rgb2grayscale",
            createModelAndInput = { ctx, height, width, batch ->
                val model = Rgb2GrayScale()
                val input = ctx.fromFloatArray<FP32, Float>(
                    shape = Shape(batch, 3, height, width),
                    dtype = FP32::class,
                    data = FloatArray(batch * 3 * height * width) { 0.5f }
                )
                ModelAndInput(model, input)
            }
        ))

        put("rgb2grayscale-matmul", ModelDescriptor(
            name = "rgb2grayscale-matmul",
            description = "RGB to Grayscale via tensor multiply (FP16)",
            functionName = "rgb2grayscale_matmul",
            createModelAndInput = { ctx, height, width, batch ->
                val model = Rgb2GrayScaleMatMul(ctx)
                val input = ctx.fromFloatArray<FP16, Float>(
                    shape = Shape(batch, 3, height, width),
                    dtype = FP16::class,
                    data = FloatArray(batch * 3 * height * width) { 0.5f }
                )
                ModelAndInput(model, input)
            }
        ))

        // Whisper-tiny.en encoder: mel spectrogram → encoder hidden states
        put("whisper-encoder", ModelDescriptor(
            name = "whisper-encoder",
            description = "Whisper-tiny.en encoder (FP32, 4 layers, 384-dim)",
            functionName = "whisper_encoder",
            createModelAndInput = { ctx, height, width, batch ->
                val metadata = WhisperModelMetadata(
                    nMels = 80,
                    nAudioCtx = 1500,
                    nAudioState = 384,
                    nAudioHead = 6,
                    nAudioLayer = 4,
                    nVocab = 51864,
                    nTextCtx = 448,
                    nTextState = 384,
                    nTextHead = 6,
                    nTextLayer = 4
                )
                val model = WhisperEncoderHloModel(metadata)
                // Mel spectrogram input: [nMels, nFrames] where nFrames = 3000 for 30s audio
                val nFrames = 3000
                val input = ctx.fromFloatArray<FP32, Float>(
                    shape = Shape(metadata.nMels, nFrames),
                    dtype = FP32::class,
                    data = FloatArray(metadata.nMels * nFrames) { 0.0f }
                )
                ModelAndInput(model, input)
            }
        ))
    }

    fun get(name: String): ModelDescriptor? = models[name]

    fun list(): Collection<ModelDescriptor> = models.values

    fun names(): Set<String> = models.keys
}
