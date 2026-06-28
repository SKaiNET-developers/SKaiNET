package sk.ainet.exec.autograd

import sk.ainet.lang.graph.DefaultGradientTape
import sk.ainet.lang.tensor.ops.DifferentiableTensorOpsRules
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Autodiff-coverage guard: every op marked `@Diff` (the KSP-generated
 * [DifferentiableTensorOpsRules.ruleNames]) must have a wired backward dispatch arm in
 * [DefaultGradientTape] (its `dispatchedOpNames`).
 *
 * The `@Diff` → generated `DifferentiableTensorOps` interface already forces a backward *formula*
 * to exist (compile error otherwise). This test closes the remaining link: that the formula is
 * actually *reachable* from the trace dispatch. It would have caught the historical bug where
 * `elu`/`leakyRelu`/`permute` had correct backward formulas that were never wired into the
 * dispatch, so their gradients were silently dropped.
 */
class AutodiffCoverageTest {

    @Test
    fun every_diff_op_has_a_wired_backward_dispatch() {
        val dispatched = DefaultGradientTape().dispatchedOpNames
        val missing = DifferentiableTensorOpsRules.ruleNames - dispatched
        assertTrue(
            missing.isEmpty(),
            "These @Diff ops have a generated backward contract but no dispatch arm in " +
                "DefaultExecutionTape.backwardDispatch (their gradients would silently drop to null): $missing",
        )
    }
}
