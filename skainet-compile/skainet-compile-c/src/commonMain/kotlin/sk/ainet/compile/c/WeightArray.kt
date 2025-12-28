package sk.ainet.compile.c

/**
 * Represents a weight or bias array for C code generation.
 * 
 * This data class encapsulates weight/bias data that needs to be serialized
 * as static const float arrays in the generated C source code.
 * 
 * @property name Variable name for the weight array in C code
 * @property values Array of weight/bias values
 * @property shape Shape of the weight tensor as an array of dimensions
 * @property isWeight True if this represents weights, false if biases
 */
public data class WeightArray(
    val name: String,
    val values: FloatArray,
    val shape: IntArray,
    val isWeight: Boolean = true
) {
    init {
        require(name.isNotBlank()) { "name cannot be blank" }
        require(values.isNotEmpty()) { "values cannot be empty" }
        require(shape.isNotEmpty()) { "shape cannot be empty" }
        require(shape.all { it > 0 }) { "All shape dimensions must be positive" }
        require(values.size == shape.reduce { acc, dim -> acc * dim }) { 
            "values size (${values.size}) must match shape product (${shape.reduce { acc, dim -> acc * dim }})" 
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as WeightArray

        if (name != other.name) return false
        if (!values.contentEquals(other.values)) return false
        if (!shape.contentEquals(other.shape)) return false
        if (isWeight != other.isWeight) return false

        return true
    }

    override fun hashCode(): Int {
        var result = name.hashCode()
        result = 31 * result + values.contentHashCode()
        result = 31 * result + shape.contentHashCode()
        result = 31 * result + isWeight.hashCode()
        return result
    }

    override fun toString(): String {
        return "WeightArray(name='$name', shape=${shape.contentToString()}, " +
                "isWeight=$isWeight, valueCount=${values.size})"
    }
}