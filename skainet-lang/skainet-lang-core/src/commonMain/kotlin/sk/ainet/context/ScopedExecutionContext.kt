package sk.ainet.context

import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.ForwardScope
import sk.ainet.lang.memory.Scope
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * [base] with a [memoryScope] — the wiring #1135 asked about (#1145).
 *
 * The mechanism existed before this class did: [ForwardScope] bump-allocates a pre-sized slab and
 * `reset()` recycles it per step; what was missing was any reader of `ExecutionContext.memoryScope`.
 * This decorator is that reader's other half: creation methods on [ExecutionContext] consult
 * `memoryScope` when it is not [Scope.Ambient], so `zeros`/`full`/`ones`/`fromFloatArray` draw
 * from the scope's slab and their tensors die at `reset()`.
 *
 * `Ambient` remains the default everywhere — nothing changes for a context that never opts in.
 * The boundary with [ExecutionContext.scratch] is deliberate: `scratch` is untyped *intra-kernel*
 * workspace inside one op invocation; `memoryScope` governs *inter-op* activation lifetime across
 * a step. They stay separate.
 */
@ExperimentalMemoryApi
public class ScopedExecutionContext(
    private val base: ExecutionContext,
    override val memoryScope: Scope,
) : ExecutionContext by base {

    override fun <T : DType, V> zeros(shape: Shape, dtype: KClass<T>): Tensor<T, V> =
        super.zeros(shape, dtype)

    override fun <T : DType, V> ones(shape: Shape, dtype: KClass<T>): Tensor<T, V> =
        super.ones(shape, dtype)

    override fun <T : DType, V> full(shape: Shape, dtype: KClass<T>, value: Number): Tensor<T, V> =
        super.full(shape, dtype, value)

    override fun <T : DType, V> fromFloatArray(shape: Shape, dtype: KClass<T>, data: FloatArray): Tensor<T, V> =
        super.fromFloatArray(shape, dtype, data)
}

/**
 * Run [block] with a [ForwardScope] of [slabFloats] floats active on this context, closing the
 * scope (and everything it handed out) afterwards. Call `reset()` on the scope between steps:
 *
 * ```kotlin
 * ctx.forwardScope(slabFloats = 1 shl 20) { scoped, scope ->
 *     while (decoding) {
 *         step(scoped)
 *         scope.reset()   // steady state: zero new slab bytes per step
 *     }
 * }
 * ```
 */
@ExperimentalMemoryApi
public inline fun <R> ExecutionContext.forwardScope(
    slabFloats: Int,
    block: (ctx: ScopedExecutionContext, scope: ForwardScope) -> R,
): R {
    val scope = ForwardScope(slabFloats, traceSink)
    return scope.use { block(ScopedExecutionContext(this, scope), scope) }
}
