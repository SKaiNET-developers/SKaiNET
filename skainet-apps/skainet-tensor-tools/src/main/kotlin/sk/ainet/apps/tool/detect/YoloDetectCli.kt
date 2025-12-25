package sk.ainet.apps.tool.detect

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.multiple
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import onnx.ModelProto
import onnx.TensorProto
import pbandk.decodeFromByteArray
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.model.dnn.yolo.Detection
import sk.ainet.lang.model.dnn.yolo.Yolo8
import sk.ainet.lang.model.dnn.yolo.YoloConfig
import sk.ainet.lang.model.dnn.yolo.YoloPreprocess
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import java.awt.Color
import java.awt.Graphics2D
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import javax.imageio.ImageIO
import kotlin.io.path.Path
import kotlin.io.path.exists
import kotlin.io.path.isRegularFile
import kotlin.math.max
import kotlin.math.min

public fun main(args: Array<String>) {
    val parser = ArgParser("skainet-onnx-detect")
    val modelPath by parser.option(ArgType.String, shortName = "m", description = "Path to ONNX model")
    val imagePaths by parser.option(
        ArgType.String,
        shortName = "i",
        description = "Input bitmap images (one or more; can be repeated)"
    ).multiple()
    val outputPath by parser.option(
        ArgType.String,
        shortName = "o",
        description = "Output JSON path"
    )
    val scoreThreshold by parser.option(
        ArgType.Double,
        shortName = "t",
        description = "Score threshold (overrides model config)"
    )
    val iouThreshold by parser.option(
        ArgType.Double,
        description = "IoU threshold for NMS (overrides model config)"
    )
    val topK by parser.option(
        ArgType.Int,
        description = "Max detections per image (overrides model config)"
    )

    parser.parse(args)

    require(!modelPath.isNullOrBlank()) { "Model path is required (-m)" }
    require(imagePaths.isNotEmpty()) { "Provide at least one input image" }

    val modelFile = Path(modelPath!!)
    require(modelFile.exists() && modelFile.isRegularFile()) { "Model not found: $modelFile" }
    val resolvedImages = imagePaths.map { Path(it) }
    resolvedImages.forEach {
        require(it.exists() && it.isRegularFile()) { "Image not found: $it" }
    }

    val output = outputPath?.let { Path(it) }
        ?: modelFile.parent?.resolve("detections.json")
        ?: Path("detections.json")

    val modelBytes = Files.readAllBytes(modelFile)
    val modelProto = ModelProto.decodeFromByteArray(modelBytes)
    val classNames = parseClassNames(modelProto)
    val scalingHints = inferScalingHints(modelProto)
    val baseConfig = if (classNames.isNotEmpty()) {
        YoloConfig(
            numClasses = classNames.size,
            classNames = classNames.values.toList()
        )
    } else {
        YoloConfig()
    }
    val config = baseConfig.copy(
        confThreshold = scoreThreshold?.toFloat() ?: baseConfig.confThreshold,
        iouThreshold = iouThreshold?.toFloat() ?: baseConfig.iouThreshold,
        maxDetections = topK ?: baseConfig.maxDetections,
        baseChannels = scalingHints.baseChannels ?: baseConfig.baseChannels,
        depthMultiple = scalingHints.depthMultiple ?: baseConfig.depthMultiple
    )
    println("Using YOLO config: baseChannels=${config.baseChannels}, depthMultiple=${"%.2f".format(config.depthMultiple)}, classes=${config.numClasses}")

    val ctx = DirectCpuExecutionContext(phase = Phase.TRAIN) // TRAIN to use batch stats in BatchNorm
    val yolo = Yolo8(config)
    val module = yolo.create(ctx)

    val initLoad = loadInitializers(modelProto, ctx)
    initLoad.skipped.forEach { println("Skipping initializer: $it") }
    val mapping = applyWeights(module, initLoad.tensors)
    validateAllParametersMapped(mapping, initLoad.skipped)

    val detections = resolvedImages.map { imagePath ->
        val prep = preprocessImage(imagePath.toFile(), config.inputSize, config.inputSize, ctx)
        val input = YoloPreprocess.fromReadyTensor(
            tensor = prep.tensor,
            originalWidth = prep.originalWidth,
            originalHeight = prep.originalHeight,
            inputSize = config.inputSize,
            padW = prep.padX,
            padH = prep.padY,
            scale = prep.scale
        )
        val preds: List<Detection> = runBlocking {
            yolo.infer(
                module = module,
                input = input,
                executionContext = ctx
            )
        }
        ImageDetections(
            image = imagePath.toString(),
            width = prep.originalWidth,
            height = prep.originalHeight,
            detections = preds.map {
                DetectionDto(
                    classId = it.classId,
                    className = it.label,
                    score = it.score,
                    box = BoxDto(it.box.x1, it.box.y1, it.box.x2, it.box.y2)
                )
            }
        )
    }

    val payload = DetectionPayload(
        model = modelFile.toString(),
        classes = classNames,
        images = detections
    )
    output.parent?.let { Files.createDirectories(it) }
    val json = Json { prettyPrint = true }.encodeToString(payload)
    Files.writeString(output, json)
    println("Wrote detections for ${detections.size} images to $output")
}

// --- Weight loading ---------------------------------------------------------

internal data class InitTensor(
    val name: String,
    val isBias: Boolean,
    val shape: List<Int>,
    val tensor: Tensor<FP32, Float>
)

private data class InitializerLoadResult(
    val tensors: List<InitTensor>,
    val skipped: List<String>
)

private fun loadInitializers(model: ModelProto, ctx: DirectCpuExecutionContext): InitializerLoadResult {
    val graph = requireNotNull(model.graph) { "Model does not contain a graph" }
    val skipped = mutableListOf<String>()
    val tensors = graph.initializer
        .filterNot { it.name.contains("running_mean", ignoreCase = true) || it.name.contains("running_var", ignoreCase = true) }
        .mapNotNull { tensor ->
            decodeInitializer(tensor, ctx)
                .onFailure { skipped += "${tensor.name}: ${it.message ?: "unknown error"}" }
                .getOrNull()
        }
    return InitializerLoadResult(tensors, skipped)
}

internal data class MappingResult(
    val mapped: Int,
    val total: Int,
    val missingParams: List<String>,
    val unusedInitializers: List<String>
)

internal fun applyWeights(module: sk.ainet.lang.nn.Module<FP32, Float>, tensors: List<InitTensor>): MappingResult {
    val params = collectParams(module)
    val used = BooleanArray(tensors.size)
    var mapped = 0
    val missing = mutableListOf<String>()
    params.forEach { param ->
        val isBiasParam = param is ModuleParameter.BiasParameter
        val pShape = param.value.shape.dimensions.toList()
        val nameLower = param.name.lowercase()
        val candidates = tensors.withIndex()
            .filter { !used[it.index] && it.value.isBias == isBiasParam && shapesCompatible(pShape, it.value.shape) }
            .sortedByDescending { matchScore(nameLower, it.value.name.lowercase()) }

        val chosen = candidates.firstOrNull()
            ?: tensors.withIndex().firstOrNull { !used[it.index] && shapesCompatible(pShape, it.value.shape) }

        if (chosen != null) {
            @Suppress("UNCHECKED_CAST")
            (param as ModuleParameter<FP32, Float>).value = chosen.value.tensor
            used[chosen.index] = true
            mapped++
        } else {
            missing += "${param.name} shape=${pShape}"
        }
    }
    val unused = tensors.withIndex().filter { !used[it.index] }.map { "${it.value.name} shape=${it.value.shape}" }
    return MappingResult(mapped = mapped, total = params.size, missingParams = missing, unusedInitializers = unused)
}

private fun matchScore(paramName: String, tensorName: String): Int {
    var score = 0
    if (tensorName.contains(paramName)) score += 3
    if (paramName.contains("weight") && tensorName.contains("weight")) score += 2
    if (paramName.contains("bias") && tensorName.contains("bias")) score += 2
    return score
}

private fun collectParams(module: sk.ainet.lang.nn.Module<FP32, Float>): List<ModuleParameter<*, *>> {
    val out = mutableListOf<ModuleParameter<*, *>>()
    fun walk(m: sk.ainet.lang.nn.Module<FP32, Float>) {
        if (m is ModuleParameters<*, *>) {
            out += m.params
        }
        m.modules.forEach { child ->
            @Suppress("UNCHECKED_CAST")
            walk(child as sk.ainet.lang.nn.Module<FP32, Float>)
        }
    }
    walk(module)
    return out
}

private fun shapesCompatible(paramShape: List<Int>, tensorShape: List<Int>): Boolean {
    if (paramShape.size != tensorShape.size) return false
    return paramShape.zip(tensorShape).all { (a, b) -> a == b }
}

internal fun validateAllParametersMapped(mapping: MappingResult, skipped: List<String>) {
    require(mapping.mapped == mapping.total) {
        buildString {
            appendLine("Only mapped ${mapping.mapped}/${mapping.total} parameters from ONNX initializers; aborting to avoid inconsistent weights.")
            appendList("Missing params", mapping.missingParams)
            appendList("Unused initializers", mapping.unusedInitializers)
            appendList("Skipped initializers", skipped)
            appendLine("Note: ONNX is treated as source-of-truth weights. If shapes or counts differ from the DSL model, update the DSL module definition to match the ONNX graph.")
        }.trim()
    }
}

private fun StringBuilder.appendList(label: String, items: List<String>, limit: Int = 10) {
    if (items.isEmpty()) return
    appendLine("$label (${items.size}, showing up to $limit):")
    items.take(limit).forEach { append(" - ").appendLine(it) }
    if (items.size > limit) {
        appendLine(" - ... and ${items.size - limit} more")
    }
}

// --- Preprocess -------------------------------------------------------------

private data class Preprocessed(
    val tensor: Tensor<FP32, Float>,
    val scale: Float,
    val padX: Int,
    val padY: Int,
    val originalWidth: Int,
    val originalHeight: Int
)

private fun preprocessImage(file: File, targetWidth: Int, targetHeight: Int, ctx: DirectCpuExecutionContext): Preprocessed {
    val original = ImageIO.read(file) ?: error("Unable to read image: ${file.absolutePath}")
    val (scaled, scale, padX, padY) = letterbox(original, targetWidth, targetHeight)
    val tensorData = FloatArray(1 * 3 * targetHeight * targetWidth)
    for (y in 0 until targetHeight) {
        for (x in 0 until targetWidth) {
            val rgb = scaled.getRGB(x, y)
            val r = ((rgb shr 16) and 0xFF) / 255f
            val g = ((rgb shr 8) and 0xFF) / 255f
            val b = (rgb and 0xFF) / 255f
            val idx = y * targetWidth + x
            tensorData[idx] = r
            tensorData[targetWidth * targetHeight + idx] = g
            tensorData[2 * targetWidth * targetHeight + idx] = b
        }
    }
    val tensor = ctx.fromFloatArray<FP32, Float>(
        shape = Shape(intArrayOf(1, 3, targetHeight, targetWidth)),
        dtype = FP32::class,
        data = tensorData
    )
    return Preprocessed(
        tensor = tensor,
        scale = scale,
        padX = padX,
        padY = padY,
        originalWidth = original.width,
        originalHeight = original.height
    )
}

private data class LetterboxResult(
    val image: BufferedImage,
    val scale: Float,
    val padX: Int,
    val padY: Int
)

private fun letterbox(image: BufferedImage, targetWidth: Int, targetHeight: Int): LetterboxResult {
    val scale = min(targetWidth.toFloat() / image.width, targetHeight.toFloat() / image.height)
    val newW = (image.width * scale).toInt()
    val newH = (image.height * scale).toInt()
    val padX = ((targetWidth - newW) / 2f).toInt()
    val padY = ((targetHeight - newH) / 2f).toInt()

    val resized = BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB)
    val g: Graphics2D = resized.createGraphics()
    g.color = Color.BLACK
    g.fillRect(0, 0, targetWidth, targetHeight)
    g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
    g.drawImage(image, padX, padY, newW, newH, null)
    g.dispose()

    return LetterboxResult(resized, scale, padX, padY)
}

// --- JSON DTOs --------------------------------------------------------------

@Serializable
internal data class DetectionPayload(
    val model: String,
    val classes: Map<Int, String>,
    val images: List<ImageDetections>
)

@Serializable
internal data class ImageDetections(
    val image: String,
    val width: Int,
    val height: Int,
    val detections: List<DetectionDto>
)

@Serializable
internal data class DetectionDto(
    val classId: Int,
    val className: String? = null,
    val score: Float,
    val box: BoxDto
)

@Serializable
internal data class BoxDto(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float
)

// --- Helpers ---------------------------------------------------------------

private fun parseClassNames(model: ModelProto): Map<Int, String> {
    val raw = model.metadataProps.firstOrNull { it.key == "names" }?.value ?: return emptyMap()
    val trimmed = raw.trim().removePrefix("{").removeSuffix("}")
    if (trimmed.isEmpty()) return emptyMap()
    val entries = trimmed.split(",")
    val result = mutableMapOf<Int, String>()
    entries.forEach { entry ->
        val parts = entry.split(":")
        if (parts.size >= 2) {
            val id = parts[0].trim().trim('\'', '"').toIntOrNull()
            val name = parts.subList(1, parts.size).joinToString(":").trim().trim('\'', '"')
            if (id != null && name.isNotEmpty()) {
                result[id] = name
            }
        }
    }
    return result.toSortedMap()
}

private data class ScalingHints(val baseChannels: Int?, val depthMultiple: Float?)

private fun inferScalingHints(model: ModelProto): ScalingHints {
    val graph = model.graph ?: return ScalingHints(null, null)
    val inits = graph.initializer
    val firstConv = inits.firstOrNull { it.name.contains("model.0.conv.weight") } ?: inits.firstOrNull()
    val baseChannels = firstConv?.dims?.firstOrNull()?.toInt()

    val stageCounts = mutableMapOf<Long, MutableSet<Long>>()
    val regex = Regex("""model\.(\d+)\.m\.(\d+)\.""")
    inits.forEach { init ->
        val match = regex.find(init.name) ?: return@forEach
        val stage = match.groupValues[1].toLong()
        val idx = match.groupValues[2].toLong()
        stageCounts.getOrPut(stage) { mutableSetOf() }.add(idx)
    }
    val counts = stageCounts.toList().sortedBy { it.first }.map { it.second.size }
    val baseDepth = listOf(3, 6, 6, 3)
    val ratios = counts.zip(baseDepth.take(counts.size)).map { (c, base) -> c.toFloat() / base }
    val depthMultiple = ratios.takeIf { it.isNotEmpty() }?.average()?.toFloat()?.coerceIn(0.2f, 3f)

    return ScalingHints(baseChannels = baseChannels, depthMultiple = depthMultiple)
}

internal fun decodeInitializer(tensor: TensorProto, ctx: DirectCpuExecutionContext): Result<InitTensor> = runCatching {
    val t = tensor.toFloatTensor(ctx) ?: error("no data for dtype=${tensor.dataType}")
    InitTensor(
        name = tensor.name,
        isBias = tensor.name.lowercase().contains("bias"),
        shape = tensor.dims.map { it.toInt() },
        tensor = t
    )
}

private fun TensorProto.toFloatTensor(ctx: DirectCpuExecutionContext): Tensor<FP32, Float>? {
    val shape = Shape(dims.map { it.toInt() }.toIntArray())
    val volume = if (shape.rank == 0) 1 else shape.volume
    val floats: FloatArray = when (TensorProto.DataType.fromValue(dataType)) {
        TensorProto.DataType.FLOAT -> when {
            floatData.isNotEmpty() -> floatData.toFloatArray()
            rawData.array.isNotEmpty() -> rawData.array.toFloatArrayLE()
            else -> FloatArray(volume) { 0f }
        }
        TensorProto.DataType.INT64 -> when {
            int64Data.isNotEmpty() -> int64Data.map { it.toFloat() }.toFloatArray()
            rawData.array.isNotEmpty() -> rawData.array.toLongArrayLE().map { it.toFloat() }.toFloatArray()
            else -> FloatArray(volume) { 0f }
        }
        TensorProto.DataType.INT32 -> when {
            int32Data.isNotEmpty() -> int32Data.map { it.toFloat() }.toFloatArray()
            rawData.array.isNotEmpty() -> rawData.array.toIntArrayLE().map { it.toFloat() }.toFloatArray()
            else -> FloatArray(volume) { 0f }
        }
        else -> return null
    }
    return ctx.fromFloatArray(
        shape = shape,
        dtype = FP32::class,
        data = floats
    )
}

private fun ByteArray.toFloatArrayLE(): FloatArray {
    val buffer = ByteBuffer.wrap(this).order(ByteOrder.LITTLE_ENDIAN)
    val out = FloatArray(buffer.remaining() / Float.SIZE_BYTES)
    var i = 0
    while (buffer.remaining() >= Float.SIZE_BYTES) {
        out[i++] = buffer.float
    }
    return out
}

private fun ByteArray.toLongArrayLE(): LongArray {
    val buffer = ByteBuffer.wrap(this).order(ByteOrder.LITTLE_ENDIAN)
    val out = LongArray(buffer.remaining() / Long.SIZE_BYTES)
    var i = 0
    while (buffer.remaining() >= Long.SIZE_BYTES) {
        out[i++] = buffer.long
    }
    return out
}

private fun ByteArray.toIntArrayLE(): IntArray {
    val buffer = ByteBuffer.wrap(this).order(ByteOrder.LITTLE_ENDIAN)
    val out = IntArray(buffer.remaining() / Int.SIZE_BYTES)
    var i = 0
    while (buffer.remaining() >= Int.SIZE_BYTES) {
        out[i++] = buffer.int
    }
    return out
}
