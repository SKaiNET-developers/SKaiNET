package sk.ainet.lang.ops

/**
 * Marks a [TensorOp] function as DARC-validated.
 *
 * A reviewer (not the original author) has gone through Document, Assess,
 * Research, and Code for this function — read the partial prose, checked the
 * math against a reference implementation, verified the citations, and
 * confirmed the runtime behaviour matches the documented contract.
 *
 * Picked up by `OperatorDocProcessor` and rendered as a badge on the
 * generated operator page plus a column in the ops coverage matrix.
 *
 * See `contributing/darc-workflow.adoc` for the exact criteria.
 *
 * @param by Validator identity. Free-form, but `"First Last <user@example.com>"`
 *   is the convention so the same string can be linked back to git history.
 * @param on ISO-8601 date the validation completed, e.g. `"2026-05-24"`.
 * @param commit Optional short SHA pinning the validated prose. Empty when
 *   the validation is not pinned to a specific revision.
 * @param referencesChecked Whether the reviewer verified that every link and
 *   citation in the partial still resolves and supports the claim it backs.
 *   Defaults to `true` because if it weren't, the validation would not have
 *   passed; set to `false` only as a deliberate documentation signal.
 */
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.SOURCE)
public annotation class DarcValidated(
    val by: String,
    val on: String,
    val commit: String = "",
    val referencesChecked: Boolean = true,
)
