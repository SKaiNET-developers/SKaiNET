package sk.ainet.io.safetensors

/**
 * Represents the metadata from a SafeTensors index file.
 *
 * @property totalSize Total size in bytes of all tensor data (from "total_size" field)
 * @property additionalFields Any additional metadata fields beyond total_size
 */
public data class SafeTensorsIndexMetadata(
    val totalSize: Long?,
    val additionalFields: Map<String, String> = emptyMap()
)

/**
 * Represents a parsed model.safetensors.index.json file for sharded SafeTensors models.
 *
 * The index file maps tensor names to their containing shard files:
 * ```json
 * {
 *   "metadata": { "total_size": 14483464192 },
 *   "weight_map": {
 *     "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
 *     "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00003.safetensors",
 *     ...
 *   }
 * }
 * ```
 *
 * @property metadata Index metadata (total_size, etc.)
 * @property weightMap Mapping from tensor name to shard filename
 */
public data class SafeTensorsIndex(
    val metadata: SafeTensorsIndexMetadata,
    val weightMap: Map<String, String>
) {
    /**
     * Unique shard filenames derived from the weight map, sorted alphabetically.
     *
     * For a model with weight_map containing files like:
     * - "model-00001-of-00003.safetensors"
     * - "model-00002-of-00003.safetensors"
     * - "model-00003-of-00003.safetensors"
     *
     * Returns them in sorted order.
     */
    public val shardFiles: List<String> by lazy {
        weightMap.values.distinct().sorted()
    }

    /**
     * Number of shards in this model.
     */
    public val shardCount: Int get() = shardFiles.size

    /**
     * Total number of tensors in this model.
     */
    public val tensorCount: Int get() = weightMap.size

    /**
     * Get the shard filename containing a specific tensor.
     *
     * @param tensorName The fully-qualified tensor name
     * @return The shard filename, or null if tensor not found
     */
    public fun getShardForTensor(tensorName: String): String? = weightMap[tensorName]

    /**
     * Get all tensor names contained in a specific shard.
     *
     * @param shardFilename The shard filename
     * @return List of tensor names in that shard
     */
    public fun getTensorsInShard(shardFilename: String): List<String> {
        return weightMap.filterValues { it == shardFilename }.keys.toList()
    }

    /**
     * Get tensor count per shard as a map.
     *
     * @return Map from shard filename to tensor count
     */
    public fun getTensorCountPerShard(): Map<String, Int> {
        return weightMap.values.groupingBy { it }.eachCount()
    }
}
