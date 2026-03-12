package sk.ainet.io.weights

/**
 * Resolves module parameter paths (from the DSL module tree) to weight tensor names
 * (from model files like GGUF or SafeTensors).
 *
 * The module tree produces paths like:
 *   "MLP/blk.0/attn/attn.q_proj.weight"
 *
 * Different model formats use different tensor naming conventions:
 *   GGUF:        "blk.0.attn_q.weight"
 *   SafeTensors: "model.layers.0.self_attn.q_proj.weight"
 *
 * Implementations of this interface translate between the two.
 */
public interface WeightNameResolver {
    /**
     * Resolve a module parameter path to the tensor name in the model file.
     *
     * @param modulePath the path from the module tree (e.g. "MLP/blk.0/attn")
     * @param paramName the parameter name (e.g. "attn.q_proj.weight")
     * @return the tensor name in the model file, or null if no mapping exists
     */
    public fun resolve(modulePath: String, paramName: String): String?
}
