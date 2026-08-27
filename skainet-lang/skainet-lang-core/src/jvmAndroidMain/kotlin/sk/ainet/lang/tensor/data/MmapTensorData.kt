package sk.ainet.lang.tensor.data

import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.DType
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Marker interface for tensor data backed by a direct FloatBuffer.
 * This enables zero-copy memory-mapped file access on the JVM.
 *
 * Backends can check for this interface to use buffer-based operations
 * instead of array-based operations.
 */
public interface FloatBufferTensorData<T : DType> : TensorData<T, Float> {
    /**
     * The underlying FloatBuffer providing direct access to tensor data.
     * For mmap-backed data, this is a view of the memory-mapped region.
     */
    public val floatBuffer: FloatBuffer
}

/**
 * Memory-mapped tensor data implementation that provides zero-copy access
 * to tensor data stored in files.
 *
 * This implementation wraps a region of a memory-mapped file, interpreting
 * the bytes as little-endian 32-bit floats. It provides both:
 * - Direct FloatBuffer access for high-performance vectorized operations
 * - Standard TensorData get/set operations for compatibility
 *
 * @param T the data type constraint extending DType
 * @param initialShape the shape of the tensor
 * @param buffer the underlying byte buffer (typically a MappedByteBuffer slice)
 */
public class MmapFloatTensorData<T : DType>(
    initialShape: Shape,
    buffer: ByteBuffer
) : FloatBufferTensorData<T> {

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = shape.computeStrides()

    /**
     * FloatBuffer view of the underlying data.
     * Uses little-endian byte order (standard for GGUF and most file formats).
     */
    override val floatBuffer: FloatBuffer = buffer
        .order(ByteOrder.LITTLE_ENDIAN)
        .asFloatBuffer()

    init {
        require(floatBuffer.capacity() >= shape.volume) {
            "Buffer capacity ${floatBuffer.capacity()} is less than tensor volume ${shape.volume}"
        }
    }

    override fun get(vararg indices: Int): Float {
        val flatIndex = calcFlatIndex(indices)
        return floatBuffer.get(flatIndex)
    }

    override fun set(vararg indices: Int, value: Float) {
        val flatIndex = calcFlatIndex(indices)
        floatBuffer.put(flatIndex, value)
    }

    private fun calcFlatIndex(indices: IntArray): Int {
        require(indices.size == shape.dimensions.size) {
            "Number of indices (${indices.size}) must match tensor dimensions (${shape.dimensions.size})"
        }

        var flatIndex = 0
        for (i in indices.indices) {
            val idx = indices[i]
            require(idx >= 0 && idx < shape.dimensions[i]) {
                "Index $idx out of bounds for dimension $i with size ${shape.dimensions[i]}"
            }
            flatIndex += idx * strides[i]
        }
        return flatIndex
    }
}

/**
 * A handle to a memory-mapped file region that can provide tensor views.
 *
 * This class manages the lifecycle of a memory-mapped file and provides
 * methods to create tensor views into specific regions of the mapped data.
 *
 * Usage:
 * ```kotlin
 * val mmap = MmapTensorSource.fromFile(file)
 * val tensor1 = mmap.tensorAt<FP32>(offset1, shape1)
 * val tensor2 = mmap.tensorAt<FP32>(offset2, shape2)
 * // tensors share the same underlying memory-mapped region
 * mmap.close() // release when done
 * ```
 *
 * @param mappedBuffer the memory-mapped buffer for the entire file or region
 */
public class MmapTensorSource(
    private val mappedBuffer: MappedByteBuffer
) : AutoCloseable {

    /**
     * Creates a tensor data view at the specified byte offset.
     *
     * @param T the data type constraint
     * @param byteOffset offset in bytes from the start of the mapped region
     * @param shape the shape of the tensor to create
     * @return MmapFloatTensorData backed by the mapped memory
     */
    public fun <T : DType> floatTensorAt(
        byteOffset: Long,
        shape: Shape
    ): MmapFloatTensorData<T> {
        val byteSize = shape.volume * 4L // 4 bytes per float
        require(byteOffset >= 0) { "Offset must be non-negative" }
        require(byteOffset + byteSize <= mappedBuffer.capacity()) {
            "Tensor region [${byteOffset}, ${byteOffset + byteSize}) exceeds buffer capacity ${mappedBuffer.capacity()}"
        }

        // Create a slice view of the mapped buffer. Deliberately not chained:
        // on the Android SDK (pre-Java-9 nio API) position/limit return
        // Buffer, not ByteBuffer, so chaining does not compile there.
        val dup = mappedBuffer.duplicate()
        dup.position(byteOffset.toInt())
        dup.limit((byteOffset + byteSize).toInt())
        val slice = dup.slice()

        return MmapFloatTensorData(shape, slice)
    }

    /**
     * A direct [ByteBuffer] slice of the mapping — `[byteOffset, byteOffset + length)`, little
     * endian, position 0. Zero-copy: reads hit the file-backed pages. This is what packed
     * (quantized) tensors are served from under mapped staging (#1189); dense F32 tensors use
     * [floatTensorAt].
     */
    public fun byteBufferAt(byteOffset: Long, length: Long): ByteBuffer {
        require(byteOffset >= 0 && length >= 0) { "byteOffset and length must be non-negative" }
        require(byteOffset + length <= mappedBuffer.capacity()) {
            "Region [$byteOffset, ${byteOffset + length}) exceeds buffer capacity ${mappedBuffer.capacity()}"
        }
        // Not chained: Android's pre-Java-9 nio signatures return Buffer (see floatTensorAt).
        val dup = mappedBuffer.duplicate()
        dup.position(byteOffset.toInt())
        dup.limit((byteOffset + length).toInt())
        return dup.slice().order(ByteOrder.LITTLE_ENDIAN)
    }

    /**
     * Force the mapped memory to be loaded into physical memory.
     * This can improve first-access performance but uses more memory.
     */
    public fun load(): MmapTensorSource {
        mappedBuffer.load()
        return this
    }

    /**
     * Check if the mapped memory is resident in physical memory.
     */
    public fun isLoaded(): Boolean = mappedBuffer.isLoaded

    override fun close() {
        // MappedByteBuffer doesn't have explicit close; relies on GC
        // But we can hint the system to release resources
        // For true release, use sun.misc.Cleaner (not portable)
    }

    public companion object {
        /**
         * Memory-map an entire file for reading.
         *
         * @param channel the file channel to map
         * @param offset starting offset in the file (typically 0)
         * @param size number of bytes to map (typically file size)
         * @return MmapTensorSource for creating tensor views
         */
        public fun fromChannel(
            channel: FileChannel,
            offset: Long = 0,
            size: Long = channel.size() - offset
        ): MmapTensorSource {
            val mapped = channel.map(FileChannel.MapMode.READ_ONLY, offset, size)
            return MmapTensorSource(mapped)
        }
    }
}
