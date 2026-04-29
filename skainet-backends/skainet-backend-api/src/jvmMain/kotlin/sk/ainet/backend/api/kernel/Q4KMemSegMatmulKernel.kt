package sk.ainet.backend.api.kernel

import java.lang.foreign.MemorySegment

/**
 * F32 input × Q4_K-packed weights matrix-vector multiply where the
 * **weight tensor is supplied as a `java.lang.foreign.MemorySegment`**
 * rather than a heap [ByteArray]. JVM-only sibling of [Q4KMatmulKernel].
 *
 * Use this kernel when the Q4_K weight bytes already live in an
 * off-heap segment — typically because they were `mmap`'d from a
 * `.gguf` / `.safetensors` file, or because they were materialized
 * into an `Arena.ofShared` segment at load time. Letting a backend
 * read those bytes directly avoids the staging copy that
 * [Q4KMatmulKernel.matmul] does on every call (heap `ByteArray` →
 * temporary off-heap segment → native).
 *
 * The block layout, scale-pair packing, and lazy-`dmin` math are
 * identical to [Q4KMatmulKernel] (canonical ggml super-block, 256
 * elements, 144 bytes/block; see that kernel's kdoc for the byte
 * map). Implementations MUST NOT mutate `input` or `weight`, MUST
 * fully write `outputDim` floats starting at `output[outputOffset]`,
 * and MAY assume no aliasing between the inputs and the output.
 *
 * Lifetime contract: the caller owns the [weight] segment's [Arena].
 * The kernel must not retain pointers past the [matmul] call return —
 * no asynchronous reads, no caching of dereferenced addresses across
 * calls. Callers in turn must keep the segment's arena alive for the
 * duration of the call.
 */
public interface Q4KMemSegMatmulKernel {
    /**
     * @param input FP32 input vector (single row), heap array.
     * @param inputOffset element offset into [input] where the row starts.
     * @param weight off-heap `MemorySegment` holding the packed Q4_K
     *   weights for the full `outputDim × inputDim` tensor in canonical
     *   block-major layout `(blockIdx * outputDim + o) * 144` bytes.
     * @param weightByteOffset byte offset into [weight] where block
     *   `(0, 0)` starts.
     * @param inputDim contraction dimension; must be a multiple of 256.
     * @param outputDim number of output cells.
     * @param output FP32 output vector, heap array.
     * @param outputOffset element offset into [output] where the row
     *   starts.
     */
    public fun matmul(
        input: FloatArray, inputOffset: Int,
        weight: MemorySegment, weightByteOffset: Long,
        inputDim: Int, outputDim: Int,
        output: FloatArray, outputOffset: Int,
    )
}
