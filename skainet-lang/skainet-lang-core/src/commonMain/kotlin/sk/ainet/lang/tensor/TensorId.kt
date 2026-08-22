package sk.ainet.lang.tensor

/**
 * Stable, human-readable identity of a tensor derived from the model structure (SKEEP-003 §4.7):
 * the module path inside the `network { }` / `dag { }` tree, the parameter name, and an optional
 * discriminator for activations (`#step=17`). The *same* id names the tensor whether it is
 * materialized or symbolic, eager or compiled, and is what loaders' `NameMap`s, graph nodes
 * (`loc("…")`) and memory-debugging tools key on.
 *
 * Ids are optional — an anonymous `a matMul b` in a notebook has none.
 *
 * Two string forms:
 * - [canonical] — dotted, for logs, `loc()` attributes and greps: `model.layers[3].attn.q_proj.weight`;
 * - [legacyPath] — the slash-separated module path used by `ModuleNode.path` / `bindPaths` and the
 *   `WeightNameResolver`s (`MLP/blk.0/attn`), so existing name-based loaders keep working.
 *
 * [parse] inverts [canonical] segment-wise; module names that themselves contain `.` (e.g. `blk.0`)
 * round-trip as a string but split into more segments than they were built from. For that reason
 * **equality and hashing are defined on [canonical]**: two ids that print the same are the same id,
 * however they were built (`TensorId.parse(id.canonical) == id` always holds).
 *
 * @property modulePath module names from the root down to the owner of the parameter
 * @property parameter the parameter's short name within its module (`weight`, `bias`, `weight_ih`)
 * @property discriminator optional suffix for activations / repeated instances
 */
public class TensorId(
    public val modulePath: List<String>,
    public val parameter: String,
    public val discriminator: String? = null,
) {
    init {
        require(parameter.isNotEmpty()) { "TensorId.parameter must not be empty" }
        require(modulePath.none { it.isEmpty() }) { "TensorId.modulePath segments must not be empty: $modulePath" }
    }

    /** `model.layers[3].attn.q_proj.weight` (+ `#discriminator`). */
    public val canonical: String
        get() = buildString {
            for (s in modulePath) { append(s); append('.') }
            append(parameter)
            if (discriminator != null) { append('#'); append(discriminator) }
        }

    /** The slash-separated module path (`MLP/blk.0/attn`), empty string for a root-level parameter. */
    public fun legacyPath(separator: String = "/"): String = modulePath.joinToString(separator)

    /** The same id with a different [discriminator] (e.g. `withDiscriminator("step=17")`). */
    public fun withDiscriminator(discriminator: String?): TensorId = TensorId(modulePath, parameter, discriminator)

    /** Id of a sub-view of this tensor, e.g. `kv.layers[3].k` → `kv.layers[3].k[1024..2048]`. */
    public fun view(range: String): TensorId = TensorId(modulePath, "$parameter[$range]", discriminator)

    override fun toString(): String = canonical

    /** Equal iff the canonical strings are equal (see the class note). */
    override fun equals(other: Any?): Boolean = other is TensorId && other.canonical == canonical

    override fun hashCode(): Int = canonical.hashCode()

    public companion object {
        /**
         * Parse a [canonical] string: the last dotted segment is the parameter, the preceding
         * segments the module path, an optional `#…` suffix the discriminator.
         * @throws IllegalArgumentException for an empty or malformed string
         */
        public fun parse(canonical: String): TensorId {
            require(canonical.isNotBlank()) { "TensorId must not be blank" }
            val hash = canonical.indexOf('#')
            val body = if (hash >= 0) canonical.substring(0, hash) else canonical
            val disc = if (hash >= 0) canonical.substring(hash + 1).ifEmpty { null } else null
            val segments = body.split('.')
            require(segments.all { it.isNotEmpty() }) { "Malformed TensorId '$canonical'" }
            return TensorId(segments.dropLast(1), segments.last(), disc)
        }

        /** Build from a slash-separated module path (as produced by `bindPaths`) and a parameter name. */
        public fun fromLegacyPath(path: String?, parameter: String, separator: String = "/"): TensorId =
            TensorId(path?.takeIf { it.isNotEmpty() }?.split(separator) ?: emptyList(), parameter)
    }
}

/**
 * A tensor that can carry a [TensorId]. The core tensor implementations implement it; `assignTensorIds`
 * sets ids through this interface and reports tensors that cannot carry one.
 */
public interface TensorIdBearer {
    public var id: TensorId?
}
