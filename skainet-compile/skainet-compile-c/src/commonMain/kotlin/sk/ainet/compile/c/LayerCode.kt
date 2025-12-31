package sk.ainet.compile.c

/**
 * Represents generated C code for a single neural network layer.
 * 
 * This data class encapsulates all information needed to generate
 * C code for a specific layer operation, including metadata about
 * the layer's input/output shapes and the actual C code fragment.
 * 
 * @property layerName Unique identifier for the layer (e.g., "dense_1", "relu_2")
 * @property operationType Type of operation (e.g., "Dense", "ReLU", "Sigmoid")
 * @property inputShape Shape of the input tensor as an array of dimensions
 * @property outputShape Shape of the output tensor as an array of dimensions
 * @property codeFragment Generated C code fragment for this layer
 */
public data class LayerCode(
    val layerName: String,
    val operationType: String,
    val inputShape: IntArray,
    val outputShape: IntArray,
    val codeFragment: String
) {
    init {
        require(layerName.isNotBlank()) { "layerName cannot be blank" }
        require(operationType.isNotBlank()) { "operationType cannot be blank" }
        require(inputShape.isNotEmpty()) { "inputShape cannot be empty" }
        require(outputShape.isNotEmpty()) { "outputShape cannot be empty" }
        require(inputShape.all { it > 0 }) { "All input dimensions must be positive" }
        require(outputShape.all { it > 0 }) { "All output dimensions must be positive" }
        require(codeFragment.isNotBlank()) { "codeFragment cannot be blank" }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as LayerCode

        if (layerName != other.layerName) return false
        if (operationType != other.operationType) return false
        if (!inputShape.contentEquals(other.inputShape)) return false
        if (!outputShape.contentEquals(other.outputShape)) return false
        if (codeFragment != other.codeFragment) return false

        return true
    }

    override fun hashCode(): Int {
        var result = layerName.hashCode()
        result = 31 * result + operationType.hashCode()
        result = 31 * result + inputShape.contentHashCode()
        result = 31 * result + outputShape.contentHashCode()
        result = 31 * result + codeFragment.hashCode()
        return result
    }

    override fun toString(): String {
        return "LayerCode(layerName='$layerName', operationType='$operationType', " +
                "inputShape=${inputShape.contentToString()}, outputShape=${outputShape.contentToString()}, " +
                "codeFragment='${codeFragment.take(50)}...')"
    }
}