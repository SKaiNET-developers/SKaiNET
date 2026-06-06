package sk.ainet.compile.minerva

/**
 * Canonical phase-one Minerva layer kinds.
 */
public enum class MinervaLayerKind {
    DENSE
}

/**
 * Supported Minerva activation functions.
 */
public enum class MinervaActivation {
    RELU,
    SIGMOID,
    TANH
}

/**
 * Role assigned to a tensor in the lowered Minerva IR.
 */
public enum class MinervaTensorRole {
    INPUT,
    WEIGHT,
    BIAS,
    INTERMEDIATE,
    OUTPUT
}

/**
 * Tensor reference used by the Minerva intermediate representation.
 */
public data class MinervaTensorRef(
    public val id: String,
    public val name: String,
    public val shape: List<Int>,
    public val dtype: String,
    public val role: MinervaTensorRole,
    public val sourceNodeId: String? = null,
    public val metadata: Map<String, String> = emptyMap()
) {
    init {
        require(id.isNotBlank()) { "tensor id cannot be blank" }
        require(name.isNotBlank()) { "tensor name cannot be blank" }
        require(shape.isNotEmpty()) { "tensor shape cannot be empty" }
        require(shape.all { it > 0 }) { "tensor shape dimensions must be positive" }
        require(dtype.isNotBlank()) { "tensor dtype cannot be blank" }
    }

    public val elementCount: Int
        get() = shape.fold(1) { acc, dim -> acc * dim }
}

/**
 * A lowered phase-one Minerva layer pattern.
 */
public data class MinervaLayer(
    public val id: String,
    public val kind: MinervaLayerKind,
    public val input: MinervaTensorRef,
    public val weights: MinervaTensorRef,
    public val bias: MinervaTensorRef? = null,
    public val output: MinervaTensorRef,
    public val activation: MinervaActivation? = null,
    public val sourceNodeIds: List<String>,
    public val metadata: Map<String, String> = emptyMap()
) {
    init {
        require(id.isNotBlank()) { "layer id cannot be blank" }
        require(sourceNodeIds.isNotEmpty()) { "layer sourceNodeIds cannot be empty" }
        require(sourceNodeIds.all { it.isNotBlank() }) { "layer sourceNodeIds cannot contain blanks" }
    }

    public val hasBias: Boolean
        get() = bias != null
}

/**
 * Backend intermediate produced after Minerva graph canonicalization.
 */
public data class MinervaIntermediate(
    public val projectName: String,
    public val target: MinervaTarget,
    public val quantization: MinervaQuantization,
    public val input: MinervaTensorRef,
    public val output: MinervaTensorRef,
    public val layers: List<MinervaLayer>,
    public val tensors: List<MinervaTensorRef>,
    public val metadata: Map<String, String> = emptyMap()
) {
    init {
        require(projectName.isNotBlank()) { "projectName cannot be blank" }
        require(layers.isNotEmpty()) { "MinervaIntermediate requires at least one layer" }
        require(tensors.isNotEmpty()) { "MinervaIntermediate requires tensor references" }
    }

    public val layerCount: Int
        get() = layers.size

    public fun requireLowered(): MinervaIntermediate = this
}

