package sk.ainet.compile.hlo.generate

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
    }

    fun get(name: String): ModelDescriptor? = models[name]

    fun list(): Collection<ModelDescriptor> = models.values

    fun names(): Set<String> = models.keys
}
