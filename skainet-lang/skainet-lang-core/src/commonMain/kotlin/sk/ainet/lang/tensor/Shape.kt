package sk.ainet.lang.tensor

public data class Shape(val dimensions: IntArray) {
    public companion object {
        public operator fun invoke(vararg dimensions: Int): Shape {
            return Shape(dimensions.copyOf())
        }

        /**
         * A shape with a name on every axis: `Shape.labeled("batch" to 2, "seq" to 16, "dim" to 64)`.
         * See [axisLabels].
         */
        public fun labeled(vararg axes: Pair<String, Int>): Shape =
            Shape(IntArray(axes.size) { axes[it].second }).withAxisLabels(*Array(axes.size) { axes[it].first })
    }

    /**
     * Optional names of the axes, one per axis, or `null` (SKEEP-003 decision 8).
     *
     * Labels are for the reader: `shape["seq"]` instead of `shape[1]` in attention code. They are
     * not part of the shape's identity: [equals], [hashCode] and `copy()` ignore them, so a
     * labeled shape equals the same extents without labels. Operations do not carry them to their
     * results; label the shape where you read it.
     */
    public var axisLabels: List<String>? = null
        private set

    /** The same extents with [labels] as axis names, one distinct, non-blank name per axis. */
    public fun withAxisLabels(vararg labels: String): Shape {
        require(labels.size == dimensions.size) { "expected ${dimensions.size} axis labels, got ${labels.size}" }
        require(labels.none { it.isBlank() }) { "axis labels must not be blank: ${labels.toList()}" }
        require(labels.toSet().size == labels.size) { "axis labels must be distinct: ${labels.toList()}" }
        return Shape(dimensions).also { it.axisLabels = labels.toList() }
    }

    /** The index of the axis named [label], or `null` if this shape has no such label. */
    public fun axisOfOrNull(label: String): Int? = axisLabels?.indexOf(label)?.takeIf { it >= 0 }

    /** The index of the axis named [label]. */
    public fun axisOf(label: String): Int = axisOfOrNull(label)
        ?: throw IllegalArgumentException("no axis labeled '$label' in ${axisLabels ?: "an unlabeled shape"}")

    /** The extent of the axis named [label]. */
    public operator fun get(label: String): Int = dimensions[axisOf(label)]

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
        val labels = axisLabels
        val dimensionsString = dimensions.indices.joinToString(separator = " x ", prefix = "[", postfix = "]") {
            if (labels == null) Dim.render(dimensions[it]) else "${labels[it]}: ${Dim.render(dimensions[it])}"
        }
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