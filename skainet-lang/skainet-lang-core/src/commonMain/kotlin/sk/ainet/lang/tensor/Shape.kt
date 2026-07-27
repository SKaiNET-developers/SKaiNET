package sk.ainet.lang.tensor

public data class Shape(val dimensions: IntArray) {
    public companion object {
        public operator fun invoke(vararg dimensions: Int): Shape {
            return Shape(dimensions.copyOf())
        }
    }

    val volume: Int
        get() {
            require(!dimensions.hasDynamic()) {
                "volume is undefined for a dynamic shape (${dimensions.joinToString(" x ", "[", "]") { Dim.render(it) }}); " +
                    "a dynamic extent has no materializable element count"
            }
            return dimensions.fold(1) { a, x -> a * x }
        }

    val rank: Int
        get() = dimensions.size

    /** True if any extent is [Dim.DYNAMIC] (unknown at compile time). */
    public fun hasDynamic(): Boolean = dimensions.hasDynamic()

    /** True if the extent on [axis] is [Dim.DYNAMIC]. */
    public fun isDynamic(axis: Int): Boolean = Dim.isDynamic(dimensions[axis])

    /** Indices of every dynamic axis (empty for a fully-static shape). */
    public val dynamicAxes: List<Int>
        get() = dimensions.indices.filter { Dim.isDynamic(dimensions[it]) }

    public fun index(indices: IntArray): Int {
        assert(
            { indices.size == dimensions.size },
            { "`indices.size` must be ${dimensions.size}: ${indices.size}" })
        return dimensions.zip(indices).fold(0) { a, x ->
            assert(
                { 0 <= x.second && x.second < x.first },
                { "Illegal index: indices = ${indices}, shape = $dimensions" })
            a * x.first + x.second
        }
    }

    public operator fun get(index: Int): Int {
        return dimensions[index]
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Shape) return false

        return dimensions.contentEquals(other.dimensions)
    }

    override fun hashCode(): Int {
        return dimensions.contentHashCode()
    }

    override fun toString(): String {
        // Render each extent via Dim (a dynamic extent shows as `?`), and omit the volume when it is
        // undefined (any dynamic extent) rather than computing a corrupt product.
        val dimensionsString = dimensions.joinToString(separator = " x ", prefix = "[", postfix = "]") { Dim.render(it) }
        val volumeString = if (dimensions.hasDynamic()) "dynamic" else volume.toString()
        return "Shape: Dimensions = $dimensionsString, Size (Volume) = $volumeString"
    }
}

internal inline fun assert(value: () -> Boolean, lazyMessage: () -> Any) {
    if (!value()) {
        val message = lazyMessage()
        throw AssertionError(message)
    }
}