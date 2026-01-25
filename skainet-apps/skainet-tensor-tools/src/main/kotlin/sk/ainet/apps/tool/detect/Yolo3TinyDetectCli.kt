package sk.ainet.apps.tool.detect

import kotlinx.cli.ArgParser
import kotlinx.cli.ArgType
import kotlinx.cli.multiple
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.Phase
import sk.ainet.lang.model.dnn.yolo.*
import sk.ainet.lang.tensor.Shape
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
import kotlin.io.path.isRegularFile
import kotlin.math.min

/**
 * CLI for YOLOv3-tiny object detection using GGUF weights.
 *
 * Usage:
 *   java -cp "lib/star" sk.ainet.apps.tool.detect.Yolo3TinyDetectCliKt
 *       -m yolov3-tiny.gguf
 *       -i image.jpg
 *       -o detections.json
 */
public fun main(args: Array<String>) {
    val parser = ArgParser("skainet-yolo3tiny-detect")
    val modelPath by parser.option(ArgType.String, shortName = "m", description = "Path to GGUF model")
    val imagePaths by parser.option(
        ArgType.String,
        shortName = "i",
        description = "Input images (one or more; can be repeated)"
    ).multiple()
    val outputPath by parser.option(
        ArgType.String,
        shortName = "o",
        description = "Output JSON path"
    )
    val scoreThreshold by parser.option(
        ArgType.Double,
        shortName = "t",
        description = "Score threshold (overrides default 0.25)"
    )
    val iouThreshold by parser.option(
        ArgType.Double,
        description = "IoU threshold for NMS (overrides default 0.45)"
    )
    val inputSize by parser.option(
        ArgType.Int,
        shortName = "s",
        description = "Input size (default: 416)"
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

    val output = outputPath?.let { Path(it) }
        ?: modelFile.parent?.resolve("detections.json")
        ?: Path("detections.json")

    // Configure model
    val config = Yolo3TinyConfig(
        numClasses = 80,
        inputSize = inputSize ?: 416,
        confThreshold = scoreThreshold?.toFloat() ?: 0.25f,
        iouThreshold = iouThreshold?.toFloat() ?: 0.45f,
        classNames = COCO_CLASSES
    )

    println("YOLOv3-tiny Detection")
    println("  Model: $modelPath")
    println("  Input size: ${config.inputSize}")
    println("  Conf threshold: ${config.confThreshold}")
    println("  IoU threshold: ${config.iouThreshold}")
    println()

    // Create model
    val ctx = DirectCpuExecutionContext(phase = Phase.TRAIN)
    val yolo = Yolo3Tiny(config)
    val module = yolo.create(ctx)

    // Load weights from GGUF
    println("Loading weights from GGUF...")
    val loadResult = GgufYoloLoader.loadWeights(module, modelPath!!, ctx)
    println(loadResult)

    if (!loadResult.success) {
        println("Warning: Not all parameters were loaded. Results may be incorrect.")
    }

    // Process images
    val detections = resolvedImages.map { imagePath ->
        println("Processing: $imagePath")
        val prep = preprocessImage(imagePath.toFile(), config.inputSize, config.inputSize, ctx)
        val input = YoloInput(
            tensor = prep.tensor,
            originalWidth = prep.originalWidth,
            originalHeight = prep.originalHeight,
            letterboxScale = prep.scale,
            padW = prep.padX,
            padH = prep.padY
        )

        val preds: List<Detection> = runBlocking {
            yolo.infer(
                module = module,
                input = input,
                executionContext = ctx
            )
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

    // Write results
    val payload = DetectionPayload(
        model = modelFile.toString(),
        classes = COCO_CLASSES.mapIndexed { idx, name -> idx to name }.toMap(),
        images = detections
    )
    output.parent?.let { Files.createDirectories(it) }
    val json = Json { prettyPrint = true }.encodeToString(DetectionPayload.serializer(), payload)
    Files.writeString(output, json)
    println()
    println("Wrote detections to $output")
}

// --- Preprocessing ---

private data class Yolo3Preprocessed(
    val tensor: sk.ainet.lang.tensor.Tensor<FP32, Float>,
    val scale: Float,
    val padX: Int,
    val padY: Int,
    val originalWidth: Int,
    val originalHeight: Int
)

private fun preprocessImage(
    file: File,
    targetWidth: Int,
    targetHeight: Int,
    ctx: DirectCpuExecutionContext
): Yolo3Preprocessed {
    val original = ImageIO.read(file) ?: error("Unable to read image: ${file.absolutePath}")
    val (scaled, scale, padX, padY) = letterbox(original, targetWidth, targetHeight)

    // Convert to NCHW tensor with values in [0, 1]
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

    return Yolo3Preprocessed(
        tensor = tensor,
        scale = scale,
        padX = padX,
        padY = padY,
        originalWidth = original.width,
        originalHeight = original.height
    )
}

private data class Yolo3LetterboxResult(
    val image: BufferedImage,
    val scale: Float,
    val padX: Int,
    val padY: Int
)

private fun letterbox(image: BufferedImage, targetWidth: Int, targetHeight: Int): Yolo3LetterboxResult {
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

    return Yolo3LetterboxResult(resized, scale, padX, padY)
}

// --- COCO Classes ---

private val COCO_CLASSES = listOf(
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
)
