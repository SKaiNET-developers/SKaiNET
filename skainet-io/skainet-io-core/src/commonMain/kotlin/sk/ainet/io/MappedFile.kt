package sk.ainet.io

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType

/**
 * A file whose bytes can be handed out as file-backed pages instead of heap copies — the MAPPED
 * half of `quantPolicy × staging` (SKEEP-003 §7, #1037).
 *
 * A dense FP32 tensor served by [denseFloats] never touches the managed heap: on Android that is
 * the difference between a model fitting under the ART cap and not (#921). Bytes that a kernel
 * still wants as an array come through [bytes], which copies out of the mapping.
 *
 * The returned tensor data stays valid after [close] on the platforms that implement this — the
 * mapping outlives the channel it came from — but treat closing as "no more tensors from this
 * file" and keep the object alive while you are still creating views.
 */
public interface MappedFile : AutoCloseable {

    /** Size of the mapped region in bytes. */
    public val sizeBytes: Long

    /** A dense FP32 tensor of [shape] over the mapping at [byteOffset] — zero heap bytes. */
    public fun <T : DType> denseFloats(byteOffset: Long, shape: Shape): TensorData<T, Float>

    /** [length] bytes copied out of the mapping at [byteOffset], for kernels that need an array. */
    public fun bytes(byteOffset: Long, length: Int): ByteArray

    /**
     * A packed quantized tensor of [shape]/[encoding] viewing the mapping at [byteOffset] — zero
     * heap bytes, blocks in canonical row-major file order (#1189). Returns `null` when this
     * platform (or this encoding) has no off-heap packed representation; callers fall back to
     * heap staging, which is the pre-#1189 behaviour for every packed tensor.
     */
    public fun packedTensor(
        byteOffset: Long,
        shape: Shape,
        encoding: sk.ainet.lang.tensor.storage.TensorEncoding,
    ): TensorData<*, *>? = null
}

/**
 * Map [filePath] for tensor access, or return `null` when this platform cannot map files (JS,
 * Wasm, 32-bit Kotlin/Native) or the file cannot be opened. Callers fall back to heap staging.
 */
public expect fun openMappedFile(filePath: String): MappedFile?
