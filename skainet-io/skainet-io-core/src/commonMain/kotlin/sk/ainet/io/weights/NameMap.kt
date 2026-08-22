package sk.ainet.io.weights

import sk.ainet.lang.tensor.TensorId

/**
 * Bidirectional mapping between a checkpoint's tensor names (GGUF `blk.3.attn_q.weight`, HF
 * `model.layers.3.self_attn.q_proj.weight`) and SKaiNET [TensorId]s (SKEEP-003 §4.7 rule 5).
 * One per checkpoint format and model family; nothing is dropped silently — [unmapped] lists what
 * a map does not know.
 *
 * The canonical SKaiNET ids for transformer checkpoints are family-neutral and GGUF-role based:
 * `model.embed_tokens.weight`, `model.norm.weight`, `model.lm_head.weight`, `model.rope_freqs.weight`,
 * and per layer `model.layers[N].attn.{q,k,v,o}_proj.{weight,bias}`, `model.layers[N].attn.{q,k}_norm.weight`,
 * `model.layers[N].mlp.{gate,up,down}_proj.weight`, `model.layers[N].{attn_norm,ffn_norm,post_attention_norm,post_ffw_norm}.weight`.
 */
public interface NameMap {
    /** Model family this map is for (`llama`, `qwen2`, `gemma3`). */
    public val family: String

    /** Checkpoint format this map reads (`gguf`, `safetensors`). */
    public val format: String

    /** The [TensorId] for a checkpoint tensor name, or `null` if this map does not know the name. */
    public fun toTensorId(checkpointName: String): TensorId?

    /** The checkpoint tensor name for [id], or `null` if the id has no tensor in this format/family. */
    public fun toCheckpointName(id: TensorId): String?

    /** Names of [checkpointNames] this map cannot translate — never dropped, always reported. */
    public fun unmapped(checkpointNames: Iterable<String>): List<String> = checkpointNames.filter { toTensorId(it) == null }

    /** Translate every name; unknown names map to `null` so callers can decide. */
    public fun toTensorIds(checkpointNames: Iterable<String>): Map<String, TensorId?> = checkpointNames.associateWith { toTensorId(it) }

    /**
     * This map as a legacy [WeightNameResolver]: `(modulePath, paramName)` in the slash form
     * (`model/layers[3]/attn`, `q_proj.weight`) → checkpoint name.
     */
    public fun asWeightNameResolver(): WeightNameResolver = WeightNameResolver { modulePath, paramName ->
        val segments = (modulePath.takeIf { it.isNotEmpty() }?.split("/") ?: emptyList())
        val parts = paramName.split(".")
        // a dotted paramName ("q_proj.weight") contributes its leading parts to the module path
        toCheckpointName(TensorId(segments + parts.dropLast(1), parts.last()))
    }
}

/** SAM-style constructor for [WeightNameResolver]. */
public fun WeightNameResolver(block: (modulePath: String, paramName: String) -> String?): WeightNameResolver =
    object : WeightNameResolver {
        override fun resolve(modulePath: String, paramName: String): String? = block(modulePath, paramName)
    }

/**
 * A [NameMap] built from role tables. Checkpoint names are either top-level (`token_embd.weight`)
 * or per-layer (`blk.3.attn_q.weight` / `model.layers.3.self_attn.q_proj.weight`); the tables map
 * the format's role spelling to the canonical id suffix and back.
 *
 * @property layerPrefix how a layer index is spelled in the checkpoint, with `{N}` as the placeholder
 *   (`blk.{N}.` for GGUF, `model.layers.{N}.` for HF)
 * @property topLevel checkpoint top-level tensor stem → canonical id stem (`token_embd` → `model.embed_tokens`)
 * @property layerRoles checkpoint per-layer role stem → canonical id suffix (`attn_q` → `attn.q_proj`)
 * @property suffixes accepted trailing parameter names (`weight`, `bias`)
 */
public class RoleTableNameMap(
    override val family: String,
    override val format: String,
    private val layerPrefix: String,
    private val topLevel: Map<String, String>,
    private val layerRoles: Map<String, String>,
    private val suffixes: Set<String> = setOf("weight", "bias"),
) : NameMap {
    private val layerRegex: Regex
    private val topLevelInverse: Map<String, String> = topLevel.entries.associate { (k, v) -> v to k }
    private val layerRolesInverse: Map<String, String> = layerRoles.entries.associate { (k, v) -> v to k }

    init {
        require("{N}" in layerPrefix) { "layerPrefix must contain {N}: '$layerPrefix'" }
        val (before, after) = layerPrefix.split("{N}", limit = 2)
        layerRegex = Regex("^" + Regex.escape(before) + "(\\d+)" + Regex.escape(after) + "(.+)\\.(" + suffixes.joinToString("|") { Regex.escape(it) } + ")$")
    }

    override fun toTensorId(checkpointName: String): TensorId? {
        layerRegex.matchEntire(checkpointName)?.let { m ->
            val layer = m.groupValues[1].toInt()
            val role = layerRoles[m.groupValues[2]] ?: return null
            val suffix = m.groupValues[3]
            return TensorId(listOf("model", "layers[$layer]") + role.split("."), suffix)
        }
        val dot = checkpointName.lastIndexOf('.')
        if (dot <= 0) return null
        val stem = checkpointName.substring(0, dot); val suffix = checkpointName.substring(dot + 1)
        if (suffix !in suffixes) return null
        val canonicalStem = topLevel[stem] ?: return null
        val parts = canonicalStem.split(".")
        return TensorId(parts, suffix)
    }

    override fun toCheckpointName(id: TensorId): String? {
        if (id.discriminator != null || id.parameter !in suffixes) return null
        val path = id.modulePath
        if (path.size >= 3 && path[0] == "model" && path[1].startsWith("layers[") && path[1].endsWith("]")) {
            val layer = path[1].removePrefix("layers[").removeSuffix("]").toIntOrNull() ?: return null
            val roleStem = layerRolesInverse[path.drop(2).joinToString(".")] ?: return null
            return layerPrefix.replace("{N}", layer.toString()) + roleStem + "." + id.parameter
        }
        val stem = topLevelInverse[path.joinToString(".")] ?: return null
        return "$stem.${id.parameter}"
    }
}
