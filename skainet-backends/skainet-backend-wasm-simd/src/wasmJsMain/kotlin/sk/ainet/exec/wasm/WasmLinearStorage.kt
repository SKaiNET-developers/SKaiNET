@file:OptIn(kotlin.js.ExperimentalWasmJsInterop::class)

package sk.ainet.exec.wasm

import sk.ainet.lang.memory.OffHeapAllocator
import sk.ainet.lang.memory.Owner
import sk.ainet.lang.memory.ScopeKind
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.StorageId
import sk.ainet.lang.memory.trace.NoopTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.TensorId
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

/**
 * The wasm tier's [Storage.OffHeap]: a region of the SIMD128 sidecar's linear memory
 * (SKEEP-007 resident activations). Bytes here are read and written by the sidecar's kernels in
 * place; the Kotlin side sees them through [copyInto]/[copyFrom] (bulk copies across the module
 * boundary), never element by element. This is the binding `PlatformStorage.offHeapAllocator`
 * receives from [WasmSimd.install], so a `ForwardScope` or `ModelScope` asking for
 * `MemoryDomain.HOST_OFFHEAP` gets sidecar memory that its `reset()`/`close()` frees.
 *
 * Lifetime. A scoped storage ([ScopeKind.FORWARD] / [ScopeKind.MODEL]) is freed exactly once, by its
 * scope through [close]. An ambient storage ([ScopeKind.AMBIENT], the default for op outputs in a
 * notebook-style forward pass) has no scope to free it, so it is registered with a
 * `FinalizationRegistry` in the sidecar glue: when this Kotlin object is collected the registry
 * frees the sidecar bytes — the same GC-managed lifetime a heap `FloatArray` has, one hop later.
 * Finalization is not prompt, which is why the ops layer also keeps resident activations under
 * [WasmSimd.residentBudgetBytes] and falls back to heap outputs above it.
 */
public class WasmLinearStorage private constructor(
    override val id: StorageId,
    private val base: Int,
    override val sizeBytes: Long,
    override val owner: Owner,
    override val debugOrigin: TensorId?,
    override val sink: TraceSink,
    private val token: JsAny?,
    private val mutable: Boolean,
) : Storage.OffHeap() {

    override val isMutable: Boolean get() = (owner as? Owner.Alias)?.parent?.isMutable ?: mutable

    /** The byte address in sidecar memory — kernels take it once per call. Throws when closed. */
    public fun pointer(): Int {
        checkAlive()
        return base
    }

    private var mirror: FloatArray? = null

    /** `true` once [heapMirror] has downloaded the bytes (diagnostics and tests). */
    public val isMirrored: Boolean get() = mirror != null

    /**
     * These bytes as FP32 on the Kotlin heap: one bulk download on the first call, kept for the
     * storage's life. It is how common code that has no sidecar kernel reads a resident tensor.
     * A sidecar kernel never asks for it. The copy and the sidecar bytes agree because ops do not
     * mutate their inputs; [writeFloat] writes both.
     */
    public fun heapMirror(): FloatArray {
        checkAlive()
        mirror?.let { return it }
        val n = (sizeBytes / 4).toInt()
        val arr = FloatArray(n)
        if (n > 0) WasmKernels.readFloats(base, arr, 0, n)
        mirror = arr
        return arr
    }

    /** Write one FP32 element, to the sidecar and to the heap mirror when there is one. */
    public fun writeFloat(index: Int, value: Float) {
        checkAlive()
        require(isMutable) { "storage $id is not mutable" }
        require(index >= 0 && (index + 1) * 4L <= sizeBytes) { "element $index outside $sizeBytes bytes" }
        mirror?.let { it[index] = value }
        WasmKernels.writeFloats(base + index * 4, floatArrayOf(value), 0, 1)
    }

    override fun slice(offsetBytes: Long, lengthBytes: Long): WasmLinearStorage {
        checkAlive()
        require(offsetBytes >= 0 && lengthBytes >= 0 && offsetBytes + lengthBytes <= sizeBytes) {
            "slice [$offsetBytes, ${offsetBytes + lengthBytes}) outside $sizeBytes bytes"
        }
        return WasmLinearStorage(StorageId.next(), base + offsetBytes.toInt(), lengthBytes, Owner.Alias(this), debugOrigin, sink, null, mutable)
    }

    override fun copyInto(dest: ByteArray, destOffset: Int, offset: Long, length: Int) {
        checkAlive()
        require(offset >= 0 && length >= 0 && offset + length <= sizeBytes) { "copyInto [$offset, ${offset + length}) outside $sizeBytes bytes" }
        WasmKernels.readBytes(base + offset.toInt(), dest, destOffset, length)
    }

    override fun copyFrom(src: ByteArray, srcOffset: Int, offset: Long, length: Int) {
        checkAlive()
        require(isMutable) { "storage $id is not mutable" }
        require(offset >= 0 && length >= 0 && offset + length <= sizeBytes) { "copyFrom [$offset, ${offset + length}) outside $sizeBytes bytes" }
        WasmKernels.writeBytes(base + offset.toInt(), src, srcOffset, length)
    }

    override fun onClose() {
        if (owner !is Owner.Owned || !WasmKernels.isReady) return
        token?.let { WasmKernels.untrackResident(it) }
        WasmKernels.freeResident(base, sizeBytes.toInt())
    }

    override fun toString(): String =
        "WasmLinear($id, $sizeBytes B @$base, $owner${debugOrigin?.let { ", $it" } ?: ""}${if (isAlive) "" else ", closed"})"

    /** The hook [WasmSimd.install] hands to `PlatformStorage.offHeapAllocator`. */
    public object Allocator : OffHeapAllocator {
        override fun allocate(bytes: Long, scope: ScopeKind, origin: TensorId?, sink: TraceSink): Storage =
            WasmLinearStorage.allocate(bytes, scope, origin, sink)

        /** Resident weights plus resident activations in the sidecar — the bytes `MemoryProbe` cannot see otherwise. */
        override fun heldBytes(): Long? =
            if (WasmKernels.isReady) WasmKernels.residentWeightBytes() + WasmKernels.residentActivationBytes() else 0L
    }

    public companion object {
        /**
         * Allocate [bytes] bytes of sidecar memory, zeroed unless [zero] is `false` (an op output that
         * is fully written next), owned by a scope of kind [scope]. Requires the sidecar to be up.
         */
        public fun allocate(
            bytes: Long,
            scope: ScopeKind = ScopeKind.AMBIENT,
            origin: TensorId? = null,
            sink: TraceSink = NoopTraceSink,
            zero: Boolean = true,
        ): WasmLinearStorage {
            require(bytes in 0..Int.MAX_VALUE.toLong()) { "sidecar storage is limited to 2 GB per allocation, requested $bytes bytes" }
            check(WasmKernels.isReady) { "WasmSimd.install() has not completed: no sidecar memory to allocate from" }
            val size = bytes.toInt()
            val ptr = WasmKernels.allocResident(size)
            if (zero) WasmKernels.zero(ptr, size)
            val token = if (scope == ScopeKind.AMBIENT) WasmKernels.trackResident(ptr, size) else null
            val s = WasmLinearStorage(StorageId.next(), ptr, bytes, Owner.Owned(scope), origin, sink, token, true)
            if (sink.isEnabled) sink.emit(TraceEvent.Allocation(s.id.value, scope, bytes, origin))
            return s
        }
    }
}

/**
 * Dense FP32 tensor data whose bytes live in the sidecar ([WasmLinearStorage]) — what the ops of
 * [WasmCpuOps] hand out, so a chain `matmul → gelu → multiply → add` never copies an activation back
 * to the Kotlin heap between kernels.
 *
 * [view] is the truth about the bytes (SKEEP-003 §4.1 façade): dense FP32 over the sidecar
 * storage, nothing copied. The ops find a resident operand by that view, not by this class (#1507).
 *
 * It is also a [FloatArrayTensorData], so every common fast path keeps working on it: the first
 * access to [buffer] (or [heapFloats], an element, or [copyToFloatArray]) downloads the tensor once
 * into the storage's heap mirror ([WasmLinearStorage.heapMirror]). This is the one data class whose
 * [heapFloats] is a copy of what [view] describes and not the same bytes: a caller that can use
 * sidecar memory asks [view] first, everything else gets the mirror. Ops do not mutate their
 * inputs, so the two agree; [set] writes both.
 */
public class WasmLinearTensorData<T : DType>(
    initialShape: Shape,
    public val storage: WasmLinearStorage,
) : FloatArrayTensorData<T> {

    override val shape: Shape = Shape(initialShape.dimensions.copyOf())
    private val strides: IntArray = IntArray(shape.rank).also { st ->
        var stride = 1
        for (i in shape.rank - 1 downTo 0) {
            st[i] = stride
            stride *= shape.dimensions[i]
        }
    }

    /** A mirror of this data's own, only when the storage is longer than the tensor. */
    private var ownMirror: FloatArray? = null

    init {
        require(storage.sizeBytes >= shape.volume.toLong() * 4) {
            "storage holds ${storage.sizeBytes} bytes, shape $shape needs ${shape.volume.toLong() * 4}"
        }
    }

    /** Dense FP32 over the sidecar bytes, zero-copy. Built once: shape and storage never change. */
    override val view: TensorView = TensorView.dense(storage, shape, FP32)

    /** `true` while the sidecar bytes are alive: kernels use [pointer] and nothing is downloaded. */
    public val isResident: Boolean get() = storage.isAlive

    /** The sidecar address of element 0. Throws when the storage is closed. */
    public fun pointer(): Int = storage.pointer()

    /** `true` once a heap mirror has been downloaded (diagnostics and tests). */
    public val isMirrored: Boolean get() = ownMirror != null || storage.isMirrored

    override val buffer: FloatArray
        get() {
            val n = shape.volume
            if (storage.sizeBytes == n.toLong() * 4) return storage.heapMirror()
            storage.checkAlive()
            ownMirror?.let { return it }
            val arr = FloatArray(n)
            if (n > 0) WasmKernels.readFloats(storage.pointer(), arr, 0, n)
            ownMirror = arr
            return arr
        }

    private fun flatIndex(indices: IntArray): Int {
        require(indices.size == shape.dimensions.size) {
            "Number of indices (${indices.size}) must match tensor dimensions (${shape.dimensions.size})"
        }
        var flat = 0
        for (i in indices.indices) {
            val idx = indices[i]
            require(idx >= 0 && idx < shape.dimensions[i]) { "Index $idx out of bounds for dimension $i with size ${shape.dimensions[i]}" }
            flat += idx * strides[i]
        }
        return flat
    }

    override fun get(vararg indices: Int): Float = buffer[flatIndex(indices)]

    override fun set(vararg indices: Int, value: Float) {
        val flat = flatIndex(indices)
        ownMirror?.let { it[flat] = value }
        storage.writeFloat(flat, value)
    }

    override fun copyToFloatArray(): FloatArray = buffer.copyOf()
}
