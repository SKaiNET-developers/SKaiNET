package sk.ainet.io.gguf

import sk.ainet.io.weights.NameMap
import sk.ainet.io.weights.TransformerNameMaps
import sk.ainet.lang.tensor.TensorId

/** The [NameMap] for this GGUF's `general.architecture`, or `null` if the family is not known. */
public fun StreamingGGUFReader.nameMap(): NameMap? =
    (fields["general.architecture"] as? String)?.let { TransformerNameMaps.Gguf.forArchitecture(it) }

/**
 * Translate every tensor of this GGUF to a [TensorId] through [map] (default: the map for the
 * file's architecture). Unknown names are kept with a `null` id — see [NameMap.unmapped].
 */
public fun StreamingGGUFReader.tensorIds(map: NameMap? = nameMap()): Map<String, TensorId?> =
    tensors.associate { it.name to map?.toTensorId(it.name) }

/**
 * This map as the legacy role-based [TensorNameMapper] (GGUF names by role and layer). Roles the
 * map cannot produce throw [IllegalStateException] — a `TensorNameMapper` has no "absent" answer.
 */
public fun NameMap.asTensorNameMapper(): TensorNameMapper = object : TensorNameMapper {
    private fun name(id: TensorId): String =
        toCheckpointName(id) ?: throw IllegalStateException("$family/$format name map has no tensor for $id")
    private fun layer(n: Int, vararg rest: String, parameter: String = "weight") =
        name(TensorId(listOf("model", "layers[$n]") + rest.toList(), parameter))

    override fun tokenEmbedding(): String = name(TensorId(listOf("model", "embed_tokens"), "weight"))
    override fun outputNorm(): String = name(TensorId(listOf("model", "norm"), "weight"))
    override fun outputWeight(): String = name(TensorId(listOf("model", "lm_head"), "weight"))
    override fun layerAttnNorm(layer: Int): String = layer(layer, "attn_norm")
    override fun layerAttnQ(layer: Int): String = layer(layer, "attn", "q_proj")
    override fun layerAttnK(layer: Int): String = layer(layer, "attn", "k_proj")
    override fun layerAttnV(layer: Int): String = layer(layer, "attn", "v_proj")
    override fun layerAttnO(layer: Int): String = layer(layer, "attn", "o_proj")
    override fun layerFfnNorm(layer: Int): String = layer(layer, "ffn_norm")
    override fun layerFfnGate(layer: Int): String = layer(layer, "mlp", "gate_proj")
    override fun layerFfnUp(layer: Int): String = layer(layer, "mlp", "up_proj")
    override fun layerFfnDown(layer: Int): String = layer(layer, "mlp", "down_proj")
}
