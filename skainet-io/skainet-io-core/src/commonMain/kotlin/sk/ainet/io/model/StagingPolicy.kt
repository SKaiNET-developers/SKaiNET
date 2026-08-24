package sk.ainet.io.model

/**
 * Where a loader puts tensor bytes on their way from the file into a tensor (SKEEP-003 §7, #1037).
 *
 * Orthogonal to [QuantPolicy], which decides *what* the values are: staging decides *where the
 * bytes live*. The two are the axes of one loader — `quantPolicy × staging` — instead of the
 * separate code paths ("streaming loader" vs "mapped weights helper") they used to be.
 */
public enum class StagingPolicy {
    /** Read tensor bytes onto the heap. The historical behaviour, and the only option in a browser. */
    HEAP,

    /**
     * Map the file and serve tensors from file-backed pages: dense FP32 tensors become zero-heap
     * views the OS pages in on demand and evicts under pressure — the difference between fitting a
     * model on a 2 GB device and not (#921, #922).
     *
     * Falls back to [HEAP] when the platform cannot map (JS, Wasm), when the source is not a file,
     * or for tensor types whose kernels still consume heap `ByteArray`s (every packed format, until
     * the packed kernels take views — #973).
     */
    MAPPED,
}
