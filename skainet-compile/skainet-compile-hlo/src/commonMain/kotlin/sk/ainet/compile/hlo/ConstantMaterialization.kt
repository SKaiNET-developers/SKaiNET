package sk.ainet.compile.hlo

import sk.ainet.lang.tensor.storage.BufferHandle
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
 * The converter does not copy bytes — it passes the [source] handle
 * through unchanged. Callers that back a handle with `mmap` (planned
 * in PR E for skainet-io-gguf and skainet-io-safetensors) get a
 * true zero-copy path all the way from the source file to the `.irpa`.
 *
 * @property scope Parameter-archive scope (`@<scope>::@<key>` in the
 *     emitted MLIR, and the scope name inside the `.irpa` container).
 * @property key Symbolic name; matches `TensorSpec.name` by convention
 *     so the archive is addressable by tensor identity.
 * @property encoding Physical [TensorEncoding] (Dense / Q4_K / Q8_0 /
 *     TurboQuant / TernaryPacked / Opaque). Preserved so the packager
 *     can blit quantized blocks verbatim instead of re-quantizing.
 * @property source [BufferHandle] backing the tensor bytes. May be
 *     an in-memory copy today; PR E replaces these with mmap windows
 *     into the source GGUF / safetensors file.
 */
public data class ExternalParameterRef(
    val scope: String,
    val key: String,
    val encoding: TensorEncoding,
    val source: BufferHandle
)
