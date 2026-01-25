package sk.ainet.apps.tool.detect

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.default
import kotlinx.cli.multiple
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import onnx.ModelProto
import pbandk.decodeFromByteArray
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.io.model.ClassNames
import sk.ainet.io.onnx.OnnxModelMetadata
import sk.ainet.io.onnx.OnnxWeightLoader
import sk.ainet.lang.model.dnn.yolo.*
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.FP32
import java.awt.Color
import java.awt.Graphics2D
import java.awt.RenderingHints
import java.awt.image.BufferedImage
import java.io.File
import java.nio.file.Files
import javax.imageio.ImageIO
import kotlin.io.path.Path
import kotlin.io.path.exists
import kotlin.io.path.extension
import kotlin.io.path.isRegularFile
import kotlin.math.min

/**
 * Unified YOLO detection CLI supporting multiple model types.
 *
 * Usage:
 *   java -cp "..." sk.ainet.apps.tool.detect.YoloDetectCliKt
 *       --type v8 -m model.onnx -i image.jpg -o detections.json
 *
 * Model types:
 *   v8       - YOLOv8 (ONNX format, default)
 *   v3-tiny  - YOLOv3-tiny (GGUF format)
 *
 * If --type is not specified, auto-detects from file extension:
 *   .onnx -> v8
 *   .gguf -> v3-tiny
 */
public fun main(args: Array<String>) {
    val parser = ArgParser("skainet-yolo-detect")
    val modelType by parser.option(
        ArgType.Choice(listOf("v8", "v3-tiny", "auto"), { it }),
        fullName = "type",
        description = "Model type: v8 (ONNX), v3-tiny (GGUF), or auto (detect from extension)"
    ).default("auto")
    val modelPath by parser.option(ArgType.String, shortName = "m", description = "Path to model file")
    val imagePaths by parser.option(
        ArgType.String,
        shortName = "i",
        description = "Input images (can be repeated)"
    ).multiple()
    val outputPath by parser.option(
        ArgType.String,
        shortName = "o",
        description = "Output JSON path"
    )
    val scoreThreshold by parser.option(
        ArgType.Double,
        shortName = "t",
        description = "Score threshold (default: 0.25)"
    )
    val iouThreshold by parser.option(
        ArgType.Double,
        description = "IoU threshold for NMS (default: 0.45)"
    )
    val topK by parser.option(
        ArgType.Int,
        description = "Max detections per image"
    )
    val inputSize by parser.option(
        ArgType.Int,
        shortName = "s",
        description = "Input size (default: 640 for v8, 416 for v3-tiny)"
    )
    val baseChannels by parser.option(
        ArgType.Int,
        description = "Base channels for YOLOv8 (n=16, s=32, m=48, l=64, x=80). Auto-detected if not specified."
    )
    val depthMultiple by parser.option(
        ArgType.Double,
        description = "Depth multiplier for YOLOv8 (n=0.33, s=0.33, m=0.67, l=1.0, x=1.0). Auto-detected if not specified."
    )

    parser.parse(args)

    require(!modelPath.isNullOrBlank()) { "Model path is required (-m)" }
    require(imagePaths.isNotEmpty()) { "Provide at least one input image (-i)" }

    val modelFile = Path(modelPath!!)
    require(modelFile.exists() && modelFile.isRegularFile()) { "Model not found: $modelFile" }
    val resolvedImages = imagePaths.map { Path(it) }
    resolvedImages.forEach {
        require(it.exists() && it.isRegularFile()) { "Image not found: $it" }
    }

    // Auto-detect model type from extension if needed
    val effectiveType = if (modelType == "auto") {
        when (modelFile.extension.lowercase()) {
            "onnx" -> "v8"
            "gguf" -> "v3-tiny"
            else -> error("Cannot auto-detect model type for extension '.${modelFile.extension}'. Use --type to specify.")
        }
    } else {
        modelType
    }

    val output = outputPath?.let { Path(it) }
        ?: modelFile.parent?.resolve("detections.json")
        ?: Path("detections.json")

    println("YOLO Detection")
    println("  Model type: $effectiveType")
    println("  Model: $modelPath")

    when (effectiveType) {
        "v8" -> runYolov8(modelFile, resolvedImages, output, scoreThreshold, iouThreshold, topK, inputSize, baseChannels, depthMultiple)
        "v3-tiny" -> runYolov3Tiny(modelFile, resolvedImages, output, scoreThreshold, iouThreshold, inputSize)
        else -> error("Unknown model type: $effectiveType")
    }
}

// ============================================================================
// YOLOv8 (ONNX)
// ============================================================================

private fun runYolov8(
    modelFile: java.nio.file.Path,
    images: List<java.nio.file.Path>,
    output: java.nio.file.Path,
    scoreThreshold: Double?,
    iouThreshold: Double?,
    topK: Int?,
    inputSize: Int?,
    baseChannelsOverride: Int?,
    depthMultipleOverride: Double?
) {
    val modelBytes = Files.readAllBytes(modelFile)
    val modelProto = ModelProto.decodeFromByteArray(modelBytes)

    // Use shared ONNX metadata parser
    val onnxMetadata = OnnxModelMetadata.from(modelProto)
    val classNames = onnxMetadata.classNames ?: emptyMap()
    println("  Producer: ${onnxMetadata.producerName ?: "unknown"}")

    val scalingHints = inferScalingHints(modelProto)
    val baseConfig = if (classNames.isNotEmpty()) {
        YoloConfig(
            numClasses = classNames.size,
            classNames = classNames.values.toList()
        )
    } else {
        // Fall back to COCO classes
        YoloConfig(
            numClasses = ClassNames.coco().size,
            classNames = ClassNames.coco()
        )
    }
    val config = baseConfig.copy(
        inputSize = inputSize ?: baseConfig.inputSize,
        confThreshold = scoreThreshold?.toFloat() ?: baseConfig.confThreshold,
        iouThreshold = iouThreshold?.toFloat() ?: baseConfig.iouThreshold,
        maxDetections = topK ?: baseConfig.maxDetections,
        baseChannels = baseChannelsOverride ?: scalingHints.baseChannels ?: baseConfig.baseChannels,
        depthMultiple = depthMultipleOverride?.toFloat() ?: scalingHints.depthMultiple ?: baseConfig.depthMultiple,
        clsMidChannels = scalingHints.clsMidChannels  // Auto-detected from ONNX
    )
    println("  Input size: ${config.inputSize}")
    println("  Base channels: ${config.baseChannels}")
    println("  Depth multiple: ${config.depthMultiple}")
    println("  Cls mid channels: ${config.resolvedClsMidChannels} (${if (scalingHints.clsMidChannels != null) "auto-detected" else "default=numClasses"})")
    println("  Conf threshold: ${config.confThreshold}")
    println("  Classes: ${config.numClasses}")
    println()

    val ctx = DirectCpuExecutionContext(phase = Phase.TRAIN)
    val yolo = Yolo8(config)
    val module = yolo.create(ctx)

    val initLoad = OnnxWeightLoader.loadInitializers(modelProto, ctx)
    initLoad.skipped.forEach { println("Skipping initializer: $it") }
    val mapping = OnnxWeightLoader.applyWeightsWithDebug(module, initLoad.tensors, debug = System.getenv("DEBUG_MAPPING") == "1")
    println("  Mapped ${mapping.mapped}/${mapping.total} parameters")
    if (mapping.unusedTensors.isNotEmpty()) {
        println("  Unused initializers: ${mapping.unusedTensors.size}")
    }
    OnnxWeightLoader.validateAllParametersMapped(mapping, initLoad.skipped)

    val detections = images.map { imagePath ->
        println("Processing: $imagePath")
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
            yolo.infer(module = module, input = input, executionContext = ctx)
        }
        println("  Found ${preds.size} detections")
        preds.forEach { det ->
            println("    - ${det.label ?: "class ${det.classId}"}: ${String.format("%.2f", det.score)} @ [${det.box.x1.toInt()}, ${det.box.y1.toInt()}, ${det.box.x2.toInt()}, ${det.box.y2.toInt()}]")
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

    writeDetections(modelFile, classNames, detections, output)
}

// ============================================================================
// YOLOv3-tiny (GGUF)
// ============================================================================

private fun runYolov3Tiny(
    modelFile: java.nio.file.Path,
    images: List<java.nio.file.Path>,
    output: java.nio.file.Path,
    scoreThreshold: Double?,
    iouThreshold: Double?,
    inputSize: Int?
) {
    // Read metadata from GGUF using KMP-compatible API
    println("Reading GGUF metadata...")
    val metadata = GgufYoloLoader.readMetadata(modelFile.toString())
    println("  Architecture: ${metadata.architecture ?: "unknown"}")

    // Use class names from GGUF if available, otherwise fall back to shared COCO classes
    val classNames = metadata.classNames ?: ClassNames.coco()
    val numClasses = metadata.numClasses ?: classNames.size
    val resolvedInputSize = inputSize ?: metadata.inputSize ?: 416

    if (metadata.classNames != null) {
        println("  Class names: from GGUF metadata (${classNames.size} classes)")
    } else {
        println("  Class names: using COCO defaults (${classNames.size} classes)")
    }

    val config = Yolo3TinyConfig(
        numClasses = numClasses,
        inputSize = resolvedInputSize,
        confThreshold = scoreThreshold?.toFloat() ?: 0.25f,
        iouThreshold = iouThreshold?.toFloat() ?: 0.45f,
        classNames = classNames
    )
    println("  Input size: ${config.inputSize}")
    println("  Conf threshold: ${config.confThreshold}")
    println()

    val ctx = DirectCpuExecutionContext(phase = Phase.TRAIN)
    val yolo = Yolo3Tiny(config)
    val module = yolo.create(ctx)

    println("Loading weights from GGUF...")
    val loadResult = GgufYoloLoader.loadWeights(module, modelFile.toString(), ctx)
    println("  Loaded ${loadResult.loaded}/${loadResult.total} parameters")
    if (!loadResult.success) {
        println("  Warning: Not all parameters loaded. Results may be incorrect.")
    }
    println()

    val detections = images.map { imagePath ->
        println("Processing: $imagePath")
        val prep = preprocessImageV3(imagePath.toFile(), config.inputSize, config.inputSize, ctx)
        val input = YoloInput(
            tensor = prep.tensor,
            originalWidth = prep.originalWidth,
            originalHeight = prep.originalHeight,
            letterboxScale = prep.scale,
            padW = prep.padX,
            padH = prep.padY
        )
        val preds: List<Detection> = runBlocking {
            yolo.infer(module = module, input = input, executionContext = ctx)
        }
        println("  Found ${preds.size} detections")
        preds.forEach { det ->
            println("    - ${det.label ?: "class ${det.classId}"}: ${String.format("%.2f", det.score)} @ [${det.box.x1.toInt()}, ${det.box.y1.toInt()}, ${det.box.x2.toInt()}, ${det.box.y2.toInt()}]")
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

    val classMap = classNames.mapIndexed { idx, name -> idx to name }.toMap()
    writeDetections(modelFile, classMap, detections, output)
}

private data class PreprocessedV3(
    val tensor: Tensor<FP32, Float>,
    val scale: Float,
    val padX: Int,
    val padY: Int,
    val originalWidth: Int,
    val originalHeight: Int
)

private fun preprocessImageV3(
    file: File,
    targetWidth: Int,
    targetHeight: Int,
    ctx: DirectCpuExecutionContext
): PreprocessedV3 {
    val original = ImageIO.read(file) ?: error("Unable to read image: ${file.absolutePath}")
    val (scaled, scale, padX, padY) = letterboxGray(original, targetWidth, targetHeight)
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
    return PreprocessedV3(tensor, scale, padX, padY, original.width, original.height)
}

private fun letterboxGray(image: BufferedImage, targetWidth: Int, targetHeight: Int): LetterboxResult {
    val scale = min(targetWidth.toFloat() / image.width, targetHeight.toFloat() / image.height)
    val newW = (image.width * scale).toInt()
    val newH = (image.height * scale).toInt()
    val padX = (targetWidth - newW) / 2
    val padY = (targetHeight - newH) / 2

    val resized = BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB)
    val g: Graphics2D = resized.createGraphics()
    g.color = Color(128, 128, 128) // Gray padding for YOLOv3
    g.fillRect(0, 0, targetWidth, targetHeight)
    g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR)
    g.drawImage(image, padX, padY, newW, newH, null)
    g.dispose()

    return LetterboxResult(resized, scale, padX, padY)
}

private fun writeDetections(
    modelFile: java.nio.file.Path,
    classNames: Map<Int, String>,
    detections: List<ImageDetections>,
    output: java.nio.file.Path
) {
    val payload = DetectionPayload(
        model = modelFile.toString(),
        classes = classNames,
        images = detections
    )
    output.parent?.let { Files.createDirectories(it) }
    val json = Json { prettyPrint = true }.encodeToString(DetectionPayload.serializer(), payload)
    Files.writeString(output, json)
    println()
    println("Wrote detections to $output")
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

private data class ScalingHints(
    val baseChannels: Int?,
    val depthMultiple: Float?,
    val clsMidChannels: Int?
)

private fun inferScalingHints(model: ModelProto): ScalingHints {
    val graph = model.graph ?: return ScalingHints(null, null, null)
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

    // Detect clsMidChannels from cv3 branch first conv output channels
    // model.22.cv3.0.0.conv.weight has shape [out_channels, in_channels, H, W]
    val cv3Conv = inits.firstOrNull { it.name == "model.22.cv3.0.0.conv.weight" }
    val clsMidChannels = cv3Conv?.dims?.firstOrNull()?.toInt()

    return ScalingHints(baseChannels = baseChannels, depthMultiple = depthMultiple, clsMidChannels = clsMidChannels)
}

