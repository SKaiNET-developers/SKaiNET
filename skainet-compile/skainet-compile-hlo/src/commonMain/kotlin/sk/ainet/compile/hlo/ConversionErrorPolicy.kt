package sk.ainet.compile.hlo

/**
 * Governs how the converter reacts when a node cannot be lowered.
 *
 * Historically every failure was emitted as an MLIR comment and conversion
 * continued, so a broken graph could produce a compute-free module that
 * still "succeeded" (empty `return`, exit 0) — see issue #1247, where an
 * operand-linkage failure at the first node cascaded through 1000+ nodes
 * silently. [STRICT] makes any conversion failure a thrown exception;
 * [LENIENT] preserves the historical comment-and-continue behavior for
 * callers that diff or inspect partially-converted modules.
 */
public enum class ConversionErrorPolicy {
    /** Any node that fails to convert aborts the conversion with an exception. Default. */
    STRICT,

    /** Failures become MLIR comments and conversion continues (pre-#1247 behavior). */
    LENIENT,
}

/**
 * A node's input operand could not be resolved to an SSA value — its
 * producer was never converted (or was converted under a different name).
 *
 * Thrown by [ConversionContext.resolveOperands] under
 * [ConversionErrorPolicy.STRICT] instead of silently dropping the operand,
 * which both hid the failure and shifted later operands into earlier
 * positional slots (issue #1247: a gather whose weight operand failed saw
 * its indices operand slide into slot 0).
 */
public class MissingOperandException(
    public val nodeId: String,
    public val opName: String,
    public val inputPort: Int,
    public val sourceNodeId: String,
    public val sourceOutputPort: Int,
) : IllegalStateException(
    "Node '$nodeId' (op '$opName') has no SSA value for input port $inputPort: " +
        "producer node '$sourceNodeId' (output port $sourceOutputPort) was never " +
        "successfully converted. Under ConversionErrorPolicy.LENIENT this operand " +
        "would be silently dropped, shifting later operands into earlier slots."
)

/**
 * A node failed to lower to StableHLO under [ConversionErrorPolicy.STRICT].
 * Carries the same diagnostic text the LENIENT mode would have emitted as
 * an MLIR comment, plus the causing exception when one was thrown.
 */
public class HloConversionException(
    message: String,
    cause: Throwable? = null,
) : IllegalStateException(message, cause)
