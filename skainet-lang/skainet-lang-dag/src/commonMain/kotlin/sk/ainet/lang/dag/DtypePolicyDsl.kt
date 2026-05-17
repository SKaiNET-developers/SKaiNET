package sk.ainet.lang.dag

import sk.ainet.lang.tensor.ops.Operation
import sk.ainet.lang.types.DTypePolicy

/**
 * Attribute key under which [DTypePolicy] is stored on a
 * [GraphNodeDefinition]. The constraint-resolution pass
 * (`DTypeConstraintResolutionPass`, W7 of #615) reads this key from
 * each node's [GraphNodeDefinition.attributes] map.
 *
 * Lives in `skainet-lang-dag` so both the DSL (this file) and the
 * compile-side pass (in `skainet-compile-opt`) agree on the
 * convention without either side importing the other.
 */
public const val DTYPE_POLICY_ATTRIBUTE_KEY: String = "dtype_policy"

/**
 * DSL extension on [DagBuilder] that records a graph op with an
 * attached [DTypePolicy]. Wraps the existing
 * [DagBuilder.op] entry point — the policy lands in the node's
 * [GraphNodeDefinition.attributes] under [DTYPE_POLICY_ATTRIBUTE_KEY].
 *
 * Usage:
 * ```kotlin
 * val mm = op(
 *     operation = matmul,
 *     inputs = listOf(input, weight),
 *     dtypePolicy = DTypePolicy.Require(BF16),
 * )
 * ```
 *
 * Equivalent (but lossier — no constant key, no type help) form
 * with the base `op(...)` builder:
 * ```kotlin
 * op(matmul, listOf(input, weight),
 *     attributes = mapOf(DTYPE_POLICY_ATTRIBUTE_KEY to DTypePolicy.Require(BF16)))
 * ```
 *
 * The DSL extension is preferred — typed, discoverable, and
 * survives renames cleanly via the constant.
 */
@DagDsl
public fun DagBuilder.op(
    operation: Operation,
    inputs: List<GraphValue<*>>,
    dtypePolicy: DTypePolicy,
    id: String = "",
    extraAttributes: Map<String, Any?> = emptyMap(),
): List<GraphValue<*>> = op(
    operation = operation,
    inputs = inputs,
    id = id,
    attributes = extraAttributes + (DTYPE_POLICY_ATTRIBUTE_KEY to dtypePolicy),
)

/**
 * Convenience accessor: extracts the [DTypePolicy] previously
 * attached to [node]'s defining-graph-node via [op]. Returns `null`
 * if no policy was attached or if the stored value isn't a
 * [DTypePolicy] (defensive — `attributes` is `Map<String, Any?>`).
 */
public fun GraphNodeDefinition.dtypePolicy(): DTypePolicy? =
    attributes[DTYPE_POLICY_ATTRIBUTE_KEY] as? DTypePolicy
