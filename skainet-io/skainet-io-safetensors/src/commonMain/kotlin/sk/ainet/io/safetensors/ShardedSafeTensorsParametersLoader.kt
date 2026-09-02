package sk.ainet.io.safetensors

import sk.ainet.context.ExecutionContext
import sk.ainet.io.ParametersLoader
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.DTypePolicy
import kotlin.reflect.KClass

/**
 * [ParametersLoader] for sharded SafeTensors checkpoints — the
 * `model.safetensors.index.json` + `model-NNNNN-of-NNNNN.safetensors` layout
 * HuggingFace uses for multi-file models (#1246).
 *
 * Sharded counterpart of [SafeTensorsParametersLoader]: the per-tensor
 * materialization and narrow-float policy handling are shared verbatim via
 * [SafeTensorsMaterializer], so both loaders produce identical tensors for
 * identical bytes.
 *
 * Reading rides [StreamingShardedSafeTensorsReader.openFromIndex]: every
 * shard file is opened eagerly and its handle stays open until the load
 * completes (the reader is closed when `load` returns). Tensors are
 * delivered in name-sorted order, regardless of which shard holds them.
 *
 * Unlike the single-file loader, unsupported-dtype failures are raised
 * *before* any tensor is delivered: a pre-scan over the (filtered) index
 * aggregates every tensor whose SafeTensors dtype cannot materialize into
 * the requested [DType] and throws one [IllegalArgumentException] listing
 * them all — a 40-shard load should not die on tensor 900 (mirrors the
 * GGUF loader's fail-fast contract, #919).
 *
 * Platform note: this loader resolves shard files by path
 * (`openRandomAccessSource`), which yields no sources on js/wasm/
 * androidNativeArm32. A provider-per-shard factory
 * (`(shardFilename) -> RandomAccessSource`) for those platforms is a
 * planned follow-up and needs an
 * `openFromParsedIndex(index, shardProvider)` companion on
 * [StreamingShardedSafeTensorsReader].
 *
 * @param indexPath Path to `model.safetensors.index.json`.
 * @param onProgress Optional progress callback (current, total, tensorName);
 *   `total` reflects the filtered tensor count.
 * @param bf16Policy How to handle `BFLOAT16` tensors (see
 *   [SafeTensorsParametersLoader]).
 * @param fp16Policy How to handle `FLOAT16` tensors.
 * @param allowPartial If true, missing shards are tolerated and only tensors
 *   from present shards are delivered; if false (default), a missing shard
 *   throws [SafeTensorsShardException.IncompleteShard].
 * @param tensorFilter Optional predicate over [ShardedTensorInfo]; tensors
 *   for which it returns false are neither materialized nor delivered, and
 *   are exempt from the fail-fast dtype pre-scan. This is the hook for
 *   family-side skip policy (size guards, name allowlists) — dtype/policy
 *   handling stays engine-side. `null` loads everything.
 */
public class ShardedSafeTensorsParametersLoader(
    private val indexPath: String,
    private val onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> },
    private val bf16Policy: Bf16LoadPolicy = NarrowFloatLoadPolicy.DEQUANT_TO_FP32,
    private val fp16Policy: NarrowFloatLoadPolicy = NarrowFloatLoadPolicy.DEQUANT_TO_FP32,
    private val allowPartial: Boolean = false,
    private val tensorFilter: ((ShardedTensorInfo) -> Boolean)? = null,
) : ParametersLoader {

    override suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ) {
        StreamingShardedSafeTensorsReader.openFromIndex(indexPath, allowPartial).use { reader ->
            val tensors = tensorFilter?.let { filter -> reader.tensors.filter(filter) }
                ?: reader.tensors

            failFastOnUnsupportedTensorTypes(tensors, dtype)

            val total = tensors.size.toLong()
            var current = 0L

            for (tensorInfo in tensors) {
                val bytes = reader.loadTensorData(tensorInfo)
                val shape = Shape(*tensorInfo.shape.map { it.toInt() }.toIntArray())

                val tensor: Tensor<T, V> = SafeTensorsMaterializer.materialize(
                    ctx = ctx,
                    dtype = dtype,
                    name = tensorInfo.name,
                    dataType = tensorInfo.dataType,
                    rawDtype = tensorInfo.dtype,
                    shape = shape,
                    bytes = bytes,
                    bf16Policy = bf16Policy,
                    fp16Policy = fp16Policy,
                )

                onTensorLoaded(tensorInfo.name, tensor)
                current++
                onProgress(current, total, tensorInfo.name)
            }
        }
    }

    public companion object {

        /**
         * Constructs a [ShardedSafeTensorsParametersLoader] from a generalised
         * [DTypePolicy]. Signature parity with
         * [SafeTensorsParametersLoader.withPolicy]; the policy → behaviour
         * mapping is identical (documented there).
         */
        public fun withPolicy(
            indexPath: String,
            policy: DTypePolicy,
            onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> },
            allowPartial: Boolean = false,
            tensorFilter: ((ShardedTensorInfo) -> Boolean)? = null,
        ): ShardedSafeTensorsParametersLoader = ShardedSafeTensorsParametersLoader(
            indexPath = indexPath,
            onProgress = onProgress,
            bf16Policy = SafeTensorsMaterializer.mapPolicyToBf16(policy),
            fp16Policy = SafeTensorsMaterializer.mapPolicyToFp16(policy),
            allowPartial = allowPartial,
            tensorFilter = tensorFilter,
        )

        /**
         * Pre-scan [tensors] and throw one aggregated error naming every
         * tensor whose dtype cannot materialize into [dtype], before any
         * tensor is delivered.
         */
        internal fun failFastOnUnsupportedTensorTypes(
            tensors: List<ShardedTensorInfo>,
            dtype: KClass<out DType>,
        ) {
            val mismatches = tensors.filter { SafeTensorsMaterializer.requiredDType(it.dataType) != dtype }
            if (mismatches.isNotEmpty()) {
                val listing = mismatches.joinToString(separator = "\n") {
                    "  - '${it.name}' (${it.dtype}, shard ${it.shardLocation}) requires " +
                        "${SafeTensorsMaterializer.requiredDType(it.dataType).simpleName}"
                }
                throw IllegalArgumentException(
                    "ShardedSafeTensorsParametersLoader: ${mismatches.size} of ${tensors.size} tensors " +
                        "cannot materialize as ${dtype.simpleName}; no tensors were delivered. " +
                        "Use tensorFilter to exclude them or load with the required dtype:\n$listing",
                )
            }
        }
    }
}
