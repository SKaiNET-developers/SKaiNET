package sk.ainet.io.gguf

import sk.ainet.io.ModelReader
import sk.ainet.io.TensorInfo
import sk.ainet.lang.tensor.data.TensorData

/**
 * **Legacy facade — use [StreamingGgufParametersLoader] for GGUF loading.**
 *
 * The pull-style `ModelReader.loadTensor(name)` contract returns a
 * raw `TensorData<*, *>` without an `ExecutionContext`. The
 * production GGUF loader ([StreamingGgufParametersLoader]) is a
 * push-style API bound to a context — it iterates every tensor in
 * the file, dispatches per source dtype (F32, F16, BF16, Q4_K,
 * Q8_0…) into the matching `TensorData` subtype with explicit
 * logical shape, and calls back into user code per tensor.
 *
 * That push-style API is the right shape for GGUF loading: GGUF
 * files store all tensor headers contiguously up front, so iterating
 * once is more efficient than seeking back into the file per
 * `loadTensor(name)` call. New consumers should construct a
 * [StreamingGgufParametersLoader] directly.
 *
 * This class is kept compiling so existing dependants don't break,
 * but `loadTensor` is intentionally a fail-fast stub — using the
 * legacy facade for actual GGUF loading silently corrupted dtype
 * metadata (no policy hook, no shape verification) and is exactly
 * the anti-pattern the dtype-policy RFC (#615) calls out.
 */
@Deprecated(
    message = "Use sk.ainet.io.gguf.StreamingGgufParametersLoader instead. " +
        "The streaming loader preserves source dtypes (Q4_K, Q8_0, etc.) as packed " +
        "TensorData subtypes and threads through DTypePolicy for fail-fast resolution.",
    replaceWith = ReplaceWith("StreamingGgufParametersLoader"),
)
public class GGUFModelReader : ModelReader {
    override val metadata: Map<String, Any> = emptyMap()
    override val tensors: Map<String, TensorInfo> = emptyMap()

    override suspend fun loadTensor(name: String): TensorData<*, *> {
        error(
            "GGUFModelReader is a legacy facade and does not load GGUF tensors. " +
                "Use StreamingGgufParametersLoader(sourceProvider).load(ctx, dtype, onTensorLoaded) " +
                "to iterate tensors with their source dtypes preserved (Q4_K, Q8_0, F32, etc.). " +
                "See contributing/dtype-model.adoc for the GGUF dtype-mapping reference.",
        )
    }

    override fun close() {
        // Nothing to close — this facade owns no resources.
    }
}
