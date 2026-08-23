package sk.ainet.bench.spike

import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.BenchmarkMode
import org.openjdk.jmh.annotations.Level
import org.openjdk.jmh.annotations.Mode
import org.openjdk.jmh.annotations.OutputTimeUnit
import org.openjdk.jmh.annotations.Param
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.Setup
import org.openjdk.jmh.annotations.State
import org.openjdk.jmh.infra.Blackhole
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout
import java.util.concurrent.TimeUnit

/**
 * SKEEP-003 Phase-2 spike (decision #6, issue #1016): what does it cost to put a `TensorView`
 * (shape + layout + storage) between a kernel and its bytes?
 *
 * A throw-away model of the proposed types — `SpikeStorage` (sealed: heap `FloatArray` or off-heap
 * `MemorySegment`) and `SpikeView` (offset/length over a storage, element `get`/`set` that
 * dispatches on the storage kind, plus the "unwrap once per call" fast path kernels are expected to
 * use). Each kernel is measured against its raw-array baseline:
 *
 * - elementwise `c = a + b` over 1 M floats — budget ≤ 3 % (decision #6);
 * - `gemv` 256 × 1024 (dot products, the memory-bound decode shape) — budget: within noise.
 *
 * Variants: `raw` (FloatArray baseline) · `viewHeapGet` (per-element get/set through the view over
 * heap storage) · `viewHeapUnwrap` (view.asHeapArray() once, then raw loop) ·
 * `viewOffHeapGet` (per-element through MemorySegment) · `viewOffHeapUnwrap` (segment fetched once,
 * element access via ValueLayout).
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
open class TensorViewSpikeBench {

    sealed interface SpikeStorage {
        val size: Int
        class Heap(val array: FloatArray) : SpikeStorage { override val size: Int get() = array.size }
        class OffHeap(val segment: MemorySegment) : SpikeStorage { override val size: Int get() = (segment.byteSize() / 4).toInt() }
    }

    /** Shape + layout (offset, contiguous) + storage; never owns bytes. */
    class SpikeView(val storage: SpikeStorage, val offset: Int, val length: Int) {
        fun get(i: Int): Float = when (val s = storage) {
            is SpikeStorage.Heap -> s.array[offset + i]
            is SpikeStorage.OffHeap -> s.segment.getAtIndex(ValueLayout.JAVA_FLOAT, (offset + i).toLong())
        }
        fun set(i: Int, v: Float) { when (val s = storage) {
            is SpikeStorage.Heap -> s.array[offset + i] = v
            is SpikeStorage.OffHeap -> s.segment.setAtIndex(ValueLayout.JAVA_FLOAT, (offset + i).toLong(), v)
        } }
        /** The fast path a kernel takes once per call: the backing array, or null for off-heap. */
        fun asHeapArray(): FloatArray? = (storage as? SpikeStorage.Heap)?.array
        fun segment(): MemorySegment? = (storage as? SpikeStorage.OffHeap)?.segment
    }

    @Param("1000000")
    var n: Int = 1_000_000

    private val rows = 256
    private val cols = 1024

    // raw
    private lateinit var a: FloatArray
    private lateinit var b: FloatArray
    private lateinit var c: FloatArray
    private lateinit var w: FloatArray
    private lateinit var x: FloatArray
    private lateinit var y: FloatArray
    // views over heap
    private lateinit var va: SpikeView
    private lateinit var vb: SpikeView
    private lateinit var vc: SpikeView
    private lateinit var vw: SpikeView
    private lateinit var vx: SpikeView
    private lateinit var vy: SpikeView
    // views over off-heap
    private lateinit var oa: SpikeView
    private lateinit var ob: SpikeView
    private lateinit var oc: SpikeView
    private lateinit var ow: SpikeView
    private lateinit var ox: SpikeView
    private lateinit var oy: SpikeView
    private val arena: Arena = Arena.ofShared()

    @Setup(Level.Trial)
    fun setup() {
        a = FloatArray(n) { (it % 97) * 0.01f }; b = FloatArray(n) { (it % 89) * 0.02f }; c = FloatArray(n)
        w = FloatArray(rows * cols) { (it % 31) * 0.001f }; x = FloatArray(cols) { (it % 17) * 0.05f }; y = FloatArray(rows)
        va = SpikeView(SpikeStorage.Heap(a), 0, n); vb = SpikeView(SpikeStorage.Heap(b), 0, n); vc = SpikeView(SpikeStorage.Heap(c), 0, n)
        vw = SpikeView(SpikeStorage.Heap(w), 0, rows * cols); vx = SpikeView(SpikeStorage.Heap(x), 0, cols); vy = SpikeView(SpikeStorage.Heap(y), 0, rows)
        fun seg(src: FloatArray): MemorySegment { val s = arena.allocate(src.size * 4L, 64); MemorySegment.copy(src, 0, s, ValueLayout.JAVA_FLOAT, 0, src.size); return s }
        oa = SpikeView(SpikeStorage.OffHeap(seg(a)), 0, n); ob = SpikeView(SpikeStorage.OffHeap(seg(b)), 0, n); oc = SpikeView(SpikeStorage.OffHeap(seg(c)), 0, n)
        ow = SpikeView(SpikeStorage.OffHeap(seg(w)), 0, rows * cols); ox = SpikeView(SpikeStorage.OffHeap(seg(x)), 0, cols); oy = SpikeView(SpikeStorage.OffHeap(seg(y)), 0, rows)
    }

    // ---------------- elementwise add, 1 M ----------------

    @Benchmark fun add_raw(bh: Blackhole) {
        val a = a; val b = b; val c = c
        for (i in 0 until n) c[i] = a[i] + b[i]
        bh.consume(c)
    }

    /** Control: raw arrays with loop-invariant offsets (what the unwrap path compiles to) — isolates the offset cost from the view. */
    @Benchmark fun add_rawOffset(bh: Blackhole) {
        val a = a; val b = b; val c = c
        val oa = va.offset; val ob = vb.offset; val oc = vc.offset
        for (i in 0 until n) c[oc + i] = a[oa + i] + b[ob + i]
        bh.consume(c)
    }

    @Benchmark fun add_viewHeapGet(bh: Blackhole) {
        val va = va; val vb = vb; val vc = vc
        for (i in 0 until n) vc.set(i, va.get(i) + vb.get(i))
        bh.consume(vc)
    }

    @Benchmark fun add_viewHeapUnwrap(bh: Blackhole) {
        val a = va.asHeapArray()!!; val b = vb.asHeapArray()!!; val c = vc.asHeapArray()!!
        val oa = va.offset; val ob = vb.offset; val oc = vc.offset
        for (i in 0 until n) c[oc + i] = a[oa + i] + b[ob + i]
        bh.consume(c)
    }

    @Benchmark fun add_viewOffHeapGet(bh: Blackhole) {
        val va = oa; val vb = ob; val vc = oc
        for (i in 0 until n) vc.set(i, va.get(i) + vb.get(i))
        bh.consume(vc)
    }

    @Benchmark fun add_viewOffHeapUnwrap(bh: Blackhole) {
        val sa = oa.segment()!!; val sb = ob.segment()!!; val sc = oc.segment()!!
        for (i in 0 until n) {
            val l = i.toLong()
            sc.setAtIndex(ValueLayout.JAVA_FLOAT, l, sa.getAtIndex(ValueLayout.JAVA_FLOAT, l) + sb.getAtIndex(ValueLayout.JAVA_FLOAT, l))
        }
        bh.consume(sc)
    }

    // ---------------- gemv 256 × 1024 ----------------

    @Benchmark fun gemv_raw(bh: Blackhole) {
        val w = w; val x = x; val y = y
        for (r in 0 until rows) {
            var acc = 0f; val base = r * cols
            for (k in 0 until cols) acc += w[base + k] * x[k]
            y[r] = acc
        }
        bh.consume(y)
    }

    @Benchmark fun gemv_viewHeapGet(bh: Blackhole) {
        val vw = vw; val vx = vx; val vy = vy
        for (r in 0 until rows) {
            var acc = 0f; val base = r * cols
            for (k in 0 until cols) acc += vw.get(base + k) * vx.get(k)
            vy.set(r, acc)
        }
        bh.consume(vy)
    }

    @Benchmark fun gemv_viewHeapUnwrap(bh: Blackhole) {
        val w = vw.asHeapArray()!!; val x = vx.asHeapArray()!!; val y = vy.asHeapArray()!!
        val ow = vw.offset; val ox = vx.offset; val oy = vy.offset
        for (r in 0 until rows) {
            var acc = 0f; val base = ow + r * cols
            for (k in 0 until cols) acc += w[base + k] * x[ox + k]
            y[oy + r] = acc
        }
        bh.consume(y)
    }

    @Benchmark fun gemv_viewOffHeapGet(bh: Blackhole) {
        val vw = ow; val vx = ox; val vy = oy
        for (r in 0 until rows) {
            var acc = 0f; val base = r * cols
            for (k in 0 until cols) acc += vw.get(base + k) * vx.get(k)
            vy.set(r, acc)
        }
        bh.consume(vy)
    }

    @Benchmark fun gemv_viewOffHeapUnwrap(bh: Blackhole) {
        val sw = ow.segment()!!; val sx = ox.segment()!!; val sy = oy.segment()!!
        for (r in 0 until rows) {
            var acc = 0f; val base = (r * cols).toLong()
            for (k in 0 until cols) acc += sw.getAtIndex(ValueLayout.JAVA_FLOAT, base + k) * sx.getAtIndex(ValueLayout.JAVA_FLOAT, k.toLong())
            sy.setAtIndex(ValueLayout.JAVA_FLOAT, r.toLong(), acc)
        }
        bh.consume(sy)
    }
}
