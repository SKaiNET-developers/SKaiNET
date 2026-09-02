package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType

/**
 * A [TensorData] that carries only a [Shape] and allocates NO backing buffer.
 *
 * Element access is never valid — there is nothing to read or write; every
 * accessor throws. Use it wherever a tensor exists purely so its shape can
 * thread through shape propagation or tracing:
 *
 * - [sk.ainet.lang.tensor.ops.VoidTensorOps] uses it for dynamic shapes,
 *   whose `-1` extent a real allocation would reject outright
 *   (`NegativeArraySizeException`), letting a dynamic KV-cache seq dim
 *   survive a decode trace.
 * - Shape-only module implementations (transformer building blocks that
 *   compute output shapes without touching data) previously each hand-rolled
 *   an anonymous `TensorData` with throwing accessors; they can construct
 *   this class instead.
 *
 * For a *static* shape whose zeros might legitimately be read, prefer
 * [sk.ainet.lang.tensor.data.TensorDataFactory.placeholder] — lazily
 * materialized zeros that behave like a dense buffer on first access.
 */
public class ShapeOnlyTensorData<T : DType, V>(override val shape: Shape) : TensorData<T, V> {
    private fun noData(): Nothing =
        error("shape-only tensor carries no data — it propagates shapes only")

    override fun get(vararg indices: Int): V = noData()
    override fun set(vararg indices: Int, value: V): Unit = noData()
    override fun copyToFloatArray(): FloatArray = noData()
}
