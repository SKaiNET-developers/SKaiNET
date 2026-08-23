package sk.ainet.backend.api.kernel

import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.Format
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * What a kernel consumes, declared rather than discovered (SKEEP-003 §0 *KernelKey*, §5.1): the op,
 * the [Format] of each operand, how each operand is laid out, and the placement it needs. The
 * dispatcher looks a key up instead of walking an `is`-ladder over `TensorData` subclasses — which
 * is what made every quantisation bug a dispatch bug (#993, #991).
 *
 * Keys are values: equal keys select the same kernel, and a key prints as something a log or an
 * `UnsupportedKernel` message can show: `matmul(F32/Dense(4B) contiguous × F32/Q4_K blocked) @host`.
 */
@ExperimentalMemoryApi
public data class KernelKey(
    val op: String,
    val operands: List<OperandKey>,
    val placement: Placement = Placement.HOST,
) {
    /** Where the operands live — host memory today; a device backend adds its own (PRD non-goal for M1). */
    public enum class Placement { HOST, DEVICE }

    override fun toString(): String =
        "$op(${operands.joinToString(" × ")})" + if (placement != Placement.HOST) " @${placement.name.lowercase()}" else " @host"

    public companion object {
        /** The key of `matmul(activation, weight)` as the two views describe themselves. */
        public fun matmul(activation: TensorView, weight: TensorView, placement: Placement = Placement.HOST): KernelKey =
            KernelKey("matmul", listOf(OperandKey.of(activation), OperandKey.of(weight)), placement)
    }
}

/** One operand of a [KernelKey]: its [Format] plus the layout class the kernel must cope with. */
@ExperimentalMemoryApi
public data class OperandKey(val format: Format, val layout: LayoutClass) {
    override fun toString(): String = "$format ${layout.name.lowercase()}"

    public companion object {
        /** Describe [view]: dense-and-gap-free is `CONTIGUOUS`, a packed layout is `BLOCKED`, anything else `STRIDED`. */
        public fun of(view: TensorView): OperandKey {
            val cls = when {
                view.layout.blocked -> LayoutClass.BLOCKED
                view.isContiguous -> LayoutClass.CONTIGUOUS
                else -> LayoutClass.STRIDED
            }
            return OperandKey(view.format, cls)
        }

        /** A dense contiguous operand of [format] — the shape kernels prefer. */
        public fun contiguous(format: Format): OperandKey = OperandKey(format, LayoutClass.CONTIGUOUS)
    }
}

/**
 * How an operand's bytes are arranged, as far as kernel selection cares: one gap-free run
 * ([CONTIGUOUS]), a strided view over a larger buffer ([STRIDED]), or block-packed ([BLOCKED]).
 * A kernel that declares `CONTIGUOUS` gets a gather adapter inserted for a `STRIDED` operand
 * (§5.1) — the adapter is visible in the trace, never hidden inside a kernel.
 */
@ExperimentalMemoryApi
public enum class LayoutClass { CONTIGUOUS, STRIDED, BLOCKED }

/** Thrown when no registered kernel and no adapter chain can serve a key; lists what is registered. */
@ExperimentalMemoryApi
public class UnsupportedKernelException(
    public val key: KernelKey,
    public val candidates: List<String>,
    message: String = "No kernel for $key" + if (candidates.isEmpty()) "" else "; registered: ${candidates.joinToString(", ")}",
) : IllegalArgumentException(message)

/** The encoding name a [KernelKey] uses for a format, matching `KernelProvider.supports`' dtype keys. */
@ExperimentalMemoryApi
public val Format.kernelEncodingName: String
    get() = when (val e = encoding) {
        is TensorEncoding.Dense -> dtype.name
        else -> e.name
    }
