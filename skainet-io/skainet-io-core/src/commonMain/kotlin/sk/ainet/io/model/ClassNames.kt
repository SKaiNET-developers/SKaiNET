package sk.ainet.io.model

/**
 * Common class name sets for object detection and classification models.
 *
 * These are standard dataset class names that can be used when model metadata
 * doesn't include class names, or for validation purposes.
 *
 * Usage:
 * ```kotlin
 * // Get COCO class names
 * val names = ClassNames.coco()
 *
 * // Look up class name by index
 * val className = names.getOrNull(5) // "bus"
 *
 * // Find class index by name
 * val classId = names.indexOf("person") // 0
 * ```
 */
public object ClassNames {

    /**
     * COCO 2017 dataset class names (80 classes).
     *
     * Used by: YOLOv3, YOLOv4, YOLOv5, YOLOv8, Faster R-CNN, Mask R-CNN, etc.
     */
    public fun coco(): List<String> = COCO_80

    /**
     * COCO 2017 class names as a map (classId -> className).
     */
    public fun cocoMap(): Map<Int, String> = COCO_80.mapIndexed { idx, name -> idx to name }.toMap()

    /**
     * ImageNet 2012 class names (1000 classes).
     * Returns null if not available (to be implemented).
     */
    public fun imagenet1k(): List<String>? = null // TODO: Add ImageNet-1K classes

    /**
     * VOC 2012 dataset class names (20 classes).
     */
    public fun voc(): List<String> = VOC_20

    /**
     * VOC 2012 class names as a map.
     */
    public fun vocMap(): Map<Int, String> = VOC_20.mapIndexed { idx, name -> idx to name }.toMap()

    // ========== COCO 80 Classes ==========

    private val COCO_80: List<String> = listOf(
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed",
        "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    )

    // ========== VOC 20 Classes ==========

    private val VOC_20: List<String> = listOf(
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    )
}
