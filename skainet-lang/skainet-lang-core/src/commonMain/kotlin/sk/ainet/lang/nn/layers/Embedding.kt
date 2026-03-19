package sk.ainet.lang.nn.layers

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.DualModule
import sk.ainet.lang.nn.topology.ModuleNode
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.nn.topology.ModuleParameters
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Slice
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.reshape
import sk.ainet.lang.tensor.slice
import sk.ainet.lang.tensor.IndexOutOfRangeException
import sk.ainet.lang.tensor.asIndices
import sk.ainet.lang.tensor.ops.TensorOps
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.Int32
import kotlin.random.Random
import kotlin.reflect.KClass

/**
 * Embedding layer as a DualModule: consumes integer index tensors (Int32) and produces floating outputs (OutT).
 * Supports optional paddingIdx which zeros the corresponding embedding row.
 */
public class Embedding<OutT : DType, V>(
    public val numEmbeddings: Int,
    public val embeddingDim: Int,
    initWeight: Tensor<OutT, V>,
    public val paddingIdx: Int? = null,
    override val name: String = "Embedding"
) : DualModule<Int32, OutT, V>(), ModuleParameters<OutT, V> {

    public companion object Companion {
        internal fun <OutT : DType, V> initWeight(
            ctx: ExecutionContext,
            dtype: KClass<OutT>,
            numEmbeddings: Int,
            embeddingDim: Int,
            mean: Float,
            std: Float,
            random: Random
        ): Tensor<OutT, V> {
            // Use randn if available via data factory
            val data = ctx.tensorDataFactory.randn<OutT, V>(Shape(numEmbeddings, embeddingDim), dtype, mean, std, random)
            return ctx.fromData(data, dtype)
        }
    }

    /** Default-initializing constructor with FP32 weights by default. */
    public constructor(
        ctx: ExecutionContext,
        dtype: KClass<OutT>,
        params: EmbeddingParams,
        name: String = "Embedding",
        mean: Float = 0f,
        std: Float = 0.1f,
        random: Random = Random.Default
    ) : this(
        numEmbeddings = params.numEmbeddings,
        embeddingDim = params.embeddingDim,
        initWeight = initWeight(ctx, dtype, params.numEmbeddings, params.embeddingDim, mean, std, random),
        paddingIdx = params.paddingIdx,
        name = name
    )

    init {
        val wShape = initWeight.shape.dimensions
        require(initWeight.rank == 2 && wShape[0] == numEmbeddings && wShape[1] == embeddingDim) {
            "Embedding($name): weight shape must be [numEmbeddings, embeddingDim]=[${numEmbeddings}, ${embeddingDim}], but was ${initWeight.shape}"
        }
        if (paddingIdx != null) {
            require(paddingIdx >= 0 && paddingIdx < numEmbeddings) {
                "Embedding($name): paddingIdx must be in [0, $numEmbeddings), was $paddingIdx"
            }
        }
    }

    override val params: List<ModuleParameter<OutT, V>> = listOf(
        ModuleParameter.WeightParameter("$name.weight", initWeight)
    )

    override val modules: List<ModuleNode> get() = emptyList()

    private fun gatherRow(ops: TensorOps, weight: Tensor<OutT, V>, index: Int): Tensor<OutT, V> {
        // Slice out a single row: initial shape [1, embeddingDim]
        val row2D = weight.slice<OutT, V>(listOf(
            Slice.Range<OutT, V>(index, index + 1),
            Slice.All<OutT, V>()
        ))
        // Apply padding zeroing if applicable, then reshape to 1D [embeddingDim]
        val row = if (paddingIdx != null && index == paddingIdx) ops.subtract(row2D, row2D) else row2D
        return row.reshape(Shape(embeddingDim))
    }

    override fun forward(input: Tensor<Int32, V>, ctx: ExecutionContext): Tensor<OutT, V> {
        val weight = (params[0] as ModuleParameter.WeightParameter<OutT, V>).value
        // Use ctx.ops when available so operations are traced in graph-recording contexts.
        // Fallback to weight.ops for backwards compatibility when ctx has no ops.
        val ops = ctx.ops
        return sk.ainet.lang.nn.hooks.withForwardHooks(ctx, this, input) {
            forwardImpl(weight, ops, input)
        }
    }

    /** Accepts any tensor and validates/coerces to indices in strict mode. Useful for legacy FP tensors. */
    public fun forwardAny(input: Tensor<out DType, V>, ctx: ExecutionContext, strict: Boolean = true): Tensor<OutT, V> {
        @Suppress("UNCHECKED_CAST")
        val idxTensor = (input as Tensor<DType, V>).asIndices(strict).t
        // best-effort: if not Int32 storage, repackage via context if provided
        val exec = ctx
        return if (idxTensor.dtype != Int32::class) {
            // create a fresh Int32 tensor copying values
            val vol = idxTensor.volume
            val buffer = IntArray(vol) { i ->
                val any = idxTensor.data[i]
                (any as Number).toInt()
            }
            val shape = idxTensor.shape
            val t: Tensor<Int32, V> = exec.fromIntArray(shape, Int32::class, buffer)
            forward(t, exec)
        } else {
            @Suppress("UNCHECKED_CAST")
            forward(idxTensor as Tensor<Int32, V>, ctx)
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun forwardImpl(weight: Tensor<OutT, V>, ops: TensorOps, input: Tensor<Int32, V>): Tensor<OutT, V> {
        if (paddingIdx != null) {
            // Row-by-row with padding zeroing for correctness
            val vol = input.volume
            val indices = IntArray(vol) { i -> (input.data[i] as Number).toInt() }
            val rows = List(vol) { i ->
                val row = gatherRow(ops, weight, indices[i])
                ops.unsqueeze(row, dim = 0) // [1, embeddingDim]
            }
            val concatenated = ops.concat(rows, dim = 0) // [vol, embeddingDim]
            return if (input.rank == 2) {
                ops.reshape(concatenated, Shape(input.shape[0], input.shape[1], embeddingDim))
            } else concatenated
        }

        // Use ops.gather for a single traceable operation (critical for graph compilation).
        val result = ops.gather(weight, input as Tensor<DType, *>, dim = 0) as Tensor<OutT, V>
        return when (input.rank) {
            1 -> result
            2 -> ops.reshape(result, Shape(input.shape[0], input.shape[1], embeddingDim))
            else -> error("Embedding($name): input shape ${input.shape} not supported; expected [L] or [N, L]")
        }
    }

    // Ergonomic overloads building tensors from arrays using provided context
    public fun forward(indices: IntArray, ctx: ExecutionContext): Tensor<OutT, V> {
        val t: Tensor<Int32, V> = ctx.fromIntArray(Shape(indices.size), Int32::class, indices)
        return forward(t, ctx)
    }

    public fun forward(indices: LongArray, ctx: ExecutionContext): Tensor<OutT, V> {
        val ints = IntArray(indices.size) { i ->
            val v = indices[i]
            if (v < 0 || v > Int.MAX_VALUE) throw IndexOutOfRangeException("Embedding($name): index value $v out of Int range")
            v.toInt()
        }
        val t: Tensor<Int32, V> = ctx.fromIntArray(Shape(ints.size), Int32::class, ints)
        return forward(t, ctx)
    }
}
