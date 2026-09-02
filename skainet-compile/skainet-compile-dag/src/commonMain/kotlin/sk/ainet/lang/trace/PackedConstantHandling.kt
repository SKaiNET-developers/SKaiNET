package sk.ainet.lang.trace

/**
 * How [TraceToGraphBuilder.finalize] treats a frozen parameter whose data is
 * packed/quantized ([sk.ainet.lang.tensor.storage.PackedBlockStorage]) and
 * therefore cannot be embedded as a float constant directly.
 *
 * Historically such tensors fell through to the "input" placeholder branch,
 * silently turning model weights into function arguments — issue #1247
 * measured a `func @gemma3n` with 190+ weight args and zero dot ops that
 * still exported with exit 0.
 */
public enum class PackedConstantHandling {
    /**
     * Throw [PackedConstantException] naming the tensor and its encoding.
     * Default: an unservable module must never be produced silently.
     */
    FAIL,

    /**
     * Dequantize the packed data to a dense FP32 constant at extraction time
     * via [sk.ainet.lang.tensor.storage.PackedBlockStorage.toFloatArray].
     * Opt-in: costs one dense copy of each packed weight (a Q4_K matrix
     * grows ~8x), which is exactly the memory class #1247 is fighting —
     * use only when the export target genuinely needs dense constants.
     */
    DEQUANTIZE,
}

/**
 * A frozen parameter with packed/quantized storage reached constant
 * extraction under [PackedConstantHandling.FAIL].
 */
public class PackedConstantException(
    public val tensorId: String,
    public val encodingName: String?,
) : IllegalStateException(
    "Frozen parameter '$tensorId' has packed storage" +
        (encodingName?.let { " (encoding $it)" } ?: "") +
        " and cannot be embedded as a float graph constant. Refusing to fall " +
        "back to a function-argument placeholder (that silently produces an " +
        "unservable module — issue #1247). Either load this weight dense, or " +
        "pass PackedConstantHandling.DEQUANTIZE to dequantize it to FP32 at " +
        "extraction (costs one dense copy per packed weight)."
)
