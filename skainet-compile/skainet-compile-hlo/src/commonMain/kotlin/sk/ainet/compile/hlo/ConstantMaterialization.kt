package sk.ainet.compile.hlo

import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * Policy governing how a constant tensor (a weight, a bias, a frozen
 * parameter) is materialized into the emitted StableHLO text.
 *
 * The intent is to decouple "what the converter sees" (a graph node
 * with values) from "how those values reach the deployed runtime"
 * (inline bytes vs external parameter archive). The seam exists so a
 * caller can flip between modes without changing the graph, and so the
 * converter does not grow a private weight-format.
 *
 * Introduced as the load-bearing decision point of the architecture
 * tracked in issue #523.
 */
public sealed interface ConstantMaterializationPolicy {

    /**
     * Every constant is written into the emitted text as
     * `stablehlo.constant dense<...>`. Matches the historical behavior
     * and is the default so existing callers are unaffected.
     */
    public data object InlineAlways : ConstantMaterializationPolicy

    /**
     * Every candidate constant — currently `tensor_constant`,
     * `dense_constant`, `parameter`, `param`, `weight`, `bias` — is
     * lifted out of the IR. The converter emits a `util.global` module
     * declaration and a `util.global.load` reference, and records an
     * [ExternalParameterRef] on the resulting module. A downstream
     * packager (e.g. the `skainet-io-iree-params` module planned in PR
     * C) turns the refs into a `.irpa` sidecar that
     * `iree-compile --iree-opt-import-parameters=<path>` resolves.
     *
     * If a candidate node has no accompanying byte source (no `values`
     * / `initial_value` list and no external handle), the converter
     * falls back to inline with a diagnostic comment — better to emit
     * working IR than to reference bytes that do not exist.
     *
     * @property scope Namespace written into the emitted
     *     `util.global.load` reference (`@<scope>::@<key>`) and into
     *     the [ExternalParameterRef]. "model" is the conventional
     *     default; callers may override per-module.
     */
    public data class ExternalAlways(val scope: String = "model") : ConstantMaterializationPolicy

    /**
     * Hybrid policy: small constants stay inline, large ones go
     * external. The threshold is measured in **logical bytes** —
     * `elementCount * bytesPerElement` computed from the output
     * [sk.ainet.lang.tensor.ops.TensorSpec] — not the MLIR text size.
     * This keeps the decision independent of downstream splat / dense
     * formatting.
     *
     * @property bytes Minimum logical size (inclusive) at which a
     *     constant is externalized.
     * @property scope Namespace for externalized constants (see
     *     [ExternalAlways.scope]).
     */
    public data class SizeThreshold(
        val bytes: Long,
        val scope: String = "model"
    ) : ConstantMaterializationPolicy
}

/**
 * Reference to a weight tensor that has been lifted out of the emitted
 * StableHLO text and moved behind an `util.global.load` reference. The
 * converter produces these; a downstream packager consumes them to
 * write an IREE parameter archive (`.irpa`).
 *
 * The converter does not copy bytes — it passes the [source] through
 * unchanged. A caller that names a file range ([ConstantSource.FileRange])
 * gets the bytes moved from the source file to the `.irpa` without a
 * heap copy.
 *
 * @property scope Parameter-archive scope (`@<scope>::@<key>` in the
 *     emitted MLIR, and the scope name inside the `.irpa` container).
 * @property key Symbolic name; matches `TensorSpec.name` by convention
 *     so the archive is addressable by tensor identity.
 * @property encoding Physical [TensorEncoding] (Dense / Q4_K / Q8_0 /
 *     TurboQuant / TernaryPacked / Opaque). Preserved so the packager
 *     can blit quantized blocks verbatim instead of re-quantizing.
 * @property source Where the tensor bytes are: an array in memory or a
 *     range of a file. See [ConstantSource].
 */
public data class ExternalParameterRef(
    val scope: String,
    val key: String,
    val encoding: TensorEncoding,
    val source: ConstantSource,
    /**
     * Physical block order of the packed bytes as a stable string (`ROW_MAJOR` /
     * `INPUT_BLOCK_MAJOR`), or `null` when unknown / not packed (#1179). Carried so a
     * kernel-feed-ordered weight can be blitted into the `.irpa` verbatim *and say so* —
     * a string rather than the storage-model enum, by the same no-import rule the
     * `TensorSpec` metadata follows.
     */
    val blockOrder: String? = null,
)

/**
 * Where the bytes of an externalized constant are. This is the compiler's own carrier:
 * `skainet-compile` does not import the storage model (`sk.ainet.lang.memory`, SKEEP-003a),
 * so a constant crosses this boundary as plain arrays or as a file range, never as a
 * `Storage` or a `TensorView`.
 *
 * A packager (for example `IrpaWriter`) writes [sizeInBytes] bytes for each source.
 */
public sealed interface ConstantSource {

    /** Number of bytes the packager writes for this source. */
    public val sizeInBytes: Long

    /** [sizeInBytes] bytes of [data], starting at [offset]. The array is read, never changed. */
    public class Bytes(
        public val data: ByteArray,
        public val offset: Int = 0,
        override val sizeInBytes: Long = (data.size - offset).toLong(),
    ) : ConstantSource

    /**
     * FP32 values, written little-endian. A constant above 2 GiB has no byte-array form,
     * so it travels as the float array it already is (#1247).
     */
    public class Floats(public val data: FloatArray) : ConstantSource {
        override val sizeInBytes: Long get() = data.size.toLong() * 4L
    }

    /**
     * [sizeInBytes] bytes of the file at [path], starting at [fileOffset]. Nothing is read
     * until the packager writes the entry. A streaming reader gives the numbers: the
     * tensor's absolute data offset and its byte size.
     */
    public class FileRange(
        public val path: String,
        public val fileOffset: Long,
        override val sizeInBytes: Long,
    ) : ConstantSource
}
