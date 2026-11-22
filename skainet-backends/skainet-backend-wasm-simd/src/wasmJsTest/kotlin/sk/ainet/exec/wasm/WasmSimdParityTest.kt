package sk.ainet.exec.wasm

import kotlinx.coroutines.test.runTest
import sk.ainet.backend.api.kernel.BitNetGemvKernel
import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.backend.api.kernel.KernelRegistry
import sk.ainet.backend.api.kernel.KernelStrictness
import sk.ainet.backend.api.kernel.NoSuchKernelException
import sk.ainet.backend.api.kernel.TernaryF32GemvKernel
import sk.ainet.backend.api.kernel.TernaryPlanesMatmulKernel
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ScopedExecutionContext
import sk.ainet.lang.memory.ForwardScope
import sk.ainet.lang.memory.PlatformStorage
import sk.ainet.lang.memory.StorageClosedException
import sk.ainet.lang.memory.plan.StorageCapabilities
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.storage.MemoryDomain
import sk.ainet.lang.memory.I8Absmax
import sk.ainet.lang.memory.Scope
import sk.ainet.lang.memory.Storage
import sk.ainet.lang.memory.TensorView
import sk.ainet.lang.memory.TernaryBlockDecoder
import sk.ainet.lang.memory.TernaryCodec
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceEvent
import sk.ainet.lang.tensor.data.BitNetB158TensorData
import sk.ainet.lang.tensor.data.BitNetPlanesTensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.exec.kernel.ScalarBf16MatmulKernel
import sk.ainet.exec.kernel.ScalarFp16MatmulKernel
import sk.ainet.exec.kernel.ScalarQ4_KMatmulKernel
import sk.ainet.exec.kernel.ScalarQ5_0MatmulKernel
import sk.ainet.exec.kernel.ScalarQ5_1MatmulKernel
import sk.ainet.exec.kernel.ScalarQ5_KMatmulKernel
import sk.ainet.exec.kernel.ScalarQ6_KMatmulKernel
import sk.ainet.exec.tensor.ops.DefaultCpuOps
import sk.ainet.lang.types.Bf16Codec
import sk.ainet.lang.types.Fp16Codec
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.sliceView
import sk.ainet.lang.tensor.data.Q4_0Quantizer
import sk.ainet.lang.tensor.data.Q8_0BlockTensorData
import sk.ainet.lang.tensor.data.Q8_0Quantizer
import sk.ainet.lang.tensor.data.StorageFloatTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.matmulWeightTransposed
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import sk.ainet.lang.nn.normalization.LayerNormalization
import sk.ainet.lang.tensor.ops.AttentionMask
import sk.ainet.lang.tensor.ops.FusedNormOps
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue

/**
 * SKEEP-007 §T1: the SIMD128 sidecar agrees with the common scalar lane on every path it takes
 * over — row-major packed matmuls (Q8_0, Q4_0, Q4_K via the pack), dense `x · Wᵀ`, the dense GEMM
 * and the three ternary kernels — and the wiring makes a freshly built context use [WasmCpuOps].
 *
 * Runs on Node (`wasmJsNodeTest`): the sidecar is loaded from the module's own resources. Lane
 * sums differ from the sequential scalar accumulation in the last bits, so comparisons are
 * relative-tolerance, as the JVM Panama and native parity tests are.
 */
private typealias PackedMatmul = (FloatArray, ByteArray, Int, Int, FloatArray) -> Unit

class WasmSimdParityTest {

    private fun pseudo(n: Int, seed: Int, scale: Float = 1f): FloatArray {
        var s = seed
        return FloatArray(n) {
            s = s * 1103515245 + 12345
            ((s ushr 16) % 2001 - 1000) / 1000f * scale
        }
    }

    private fun assertClose(expected: FloatArray, actual: FloatArray, what: String, rel: Float = 1e-5f) {
        assertEquals(expected.size, actual.size, "$what size")
        var maxAbs = 0f
        for (v in expected) maxAbs = maxOf(maxAbs, abs(v))
        val tol = maxAbs * rel + 1e-6f
        for (i in expected.indices) {
            assertTrue(abs(expected[i] - actual[i]) <= tol, "$what[$i]: ${actual[i]} vs ${expected[i]} (tol $tol)")
        }
    }

    @AfterTest
    fun tearDown() {
        WasmSimd.uninstall()
    }

    private fun heapFloats(view: TensorView): FloatArray = checkNotNull((view.storage as Storage.Heap).floats)

    @Suppress("UNCHECKED_CAST")
    private fun q8(ctx: DirectCpuExecutionContext, n: Int, k: Int, seed: Int): Tensor<FP32, Float> =
        ctx.fromData(Q8_0Quantizer.quantize(pseudo(n * k, seed, 2f), Shape(n, k)) as TensorData<FP32, Float>, FP32::class)

    @Suppress("UNCHECKED_CAST")
    private fun q4(ctx: DirectCpuExecutionContext, n: Int, k: Int, seed: Int): Tensor<FP32, Float> =
        ctx.fromData(Q4_0Quantizer.quantize(pseudo(n * k, seed, 2f), Shape(n, k)) as TensorData<FP32, Float>, FP32::class)

    @Test
    fun `install swaps the ops and the sidecar reports a SIMD mode`() = runTest {
        val before = DirectCpuExecutionContext()
        assertIs<DefaultCpuOps>(before.ops)
        assertEquals("none", WasmKernels.features().mode)
        WasmSimd.install()
        assertTrue(WasmSimd.isInstalled)
        assertTrue(WasmKernels.mode.startsWith("simd128-"), WasmKernels.mode)
        val after = DirectCpuExecutionContext()
        assertIs<WasmCpuOps>(after.ops)
        val f = WasmSimd.features()
        assertEquals(WasmKernels.mode, f.mode)
        assertTrue(f.simd128)
        assertEquals(f.mode == "simd128-mt", f.threads)
        assertTrue(f.hardwareConcurrency >= 1, "cores: ${f.hardwareConcurrency}")
    }

    @Test
    fun `BF16 and FP16 weights run on the sidecar and match the scalar kernels`() = runTest {
        WasmSimd.install()
        val provider = checkNotNull(KernelRegistry.bestAvailable())
        assertEquals("wasm-simd128", provider.name)
        val m = 7; val n = 33; val k = 40
        val a = pseudo(m * k, 21)
        val wf = pseudo(k * n, 23)
        val bf16 = ByteArray(k * n * 2)
        val fp16 = ByteArray(k * n * 2)
        for (i in 0 until k * n) {
            val b = Bf16Codec.encode(wf[i]); bf16[i * 2] = (b and 0xFF).toByte(); bf16[i * 2 + 1] = ((b ushr 8) and 0xFF).toByte()
            val h = Fp16Codec.encode(wf[i]); fp16[i * 2] = (h and 0xFF).toByte(); fp16[i * 2 + 1] = ((h ushr 8) and 0xFF).toByte()
        }
        val refB = FloatArray(m * n); val refH = FloatArray(m * n)
        ScalarBf16MatmulKernel.matmul(a, 0, k, bf16, 0, n * 2, refB, 0, n, m, n, k)
        ScalarFp16MatmulKernel.matmul(a, 0, k, fp16, 0, n * 2, refH, 0, n, m, n, k)
        val gotB = FloatArray(m * n); val gotH = FloatArray(m * n)
        checkNotNull(provider.matmulBf16()).matmul(a, 0, k, bf16, 0, n * 2, gotB, 0, n, m, n, k)
        checkNotNull(provider.matmulFp16()).matmul(a, 0, k, fp16, 0, n * 2, gotH, 0, n, m, n, k)
        assertClose(refB, gotB, "bf16")
        assertClose(refH, gotH, "fp16")
        // A contiguous weight is resident after the first call; a strided window is staged per call.
        assertTrue(WasmKernels.residentWeightBytes() >= 2L * k * n * 2)
        val sub = FloatArray(m * 8)
        val refSub = FloatArray(m * 8)
        ScalarBf16MatmulKernel.matmul(a, 0, k, bf16, 4, n * 2, refSub, 0, 8, m, 8, k)
        checkNotNull(provider.matmulBf16()).matmul(a, 0, k, bf16, 4, n * 2, sub, 0, 8, m, 8, k)
        assertClose(refSub, sub, "bf16 window")
    }

    @Test
    fun `strict mode is honoured on wasm`() = runTest {
        KernelStrictness.overrideEnabled = true
        try {
            assertTrue(KernelStrictness.isEnabled())
            val ctx = DirectCpuExecutionContext()
            // Packed weight, no pack installed and the reference fallback forbidden: fail fast.
            val w = q8(ctx, 64, 96, 5)
            val x = ctx.fromFloatArray<FP32, Float>(Shape(2, 96), FP32::class, pseudo(2 * 96, 1))
            KernelDispatch.clearForTesting()
            // A canonical row-major Q8_0 view has no exact scalar kernel (those are keyed input-block-major)
            // and prepack is off by default here, so this is the reference-fallback site.
            val out = sk.ainet.lang.memory.TensorView.dense(sk.ainet.lang.memory.Storage.Heap.floats(2 * 64), Shape(2, 64), sk.ainet.lang.types.FP32)
            assertFailsWith<NoSuchKernelException> {
                KernelDispatch.matmul(x.data.view!!, w.data.view!!, out, sk.ainet.lang.memory.Scope.Ambient)
            }
        } finally {
            KernelStrictness.overrideEnabled = null
        }
        assertEquals(false, KernelStrictness.isEnabled())
    }

    /**
     * Random packed bytes with the FP16 scale slots forced to 1.0 (0x3C00) so magnitudes stay
     * finite — the fixture the JVM native parity tests use. [exact] formats accumulate in FP32 on
     * both sides and agree to lane-sum tolerance; the K-quant int8 kernels (Q4_K, Q6_K) quantize the
     * activation to Q8 first, so they are gated on an aggregate relative RMS error like the JVM tests.
     */
    private fun packedParity(
        what: String, blockSize: Int, bytesPerBlock: Int, scaleSlots: IntArray, inputDim: Int, outputDim: Int,
        exact: Boolean, scalar: PackedMatmul, wasm: PackedMatmul,
    ) {
        val numBlocks = (inputDim / blockSize) * outputDim
        val rng = Random(what.hashCode())
        val packed = ByteArray(numBlocks * bytesPerBlock).also { rng.nextBytes(it) }
        for (block in 0 until numBlocks) {
            for (slot in scaleSlots) {
                packed[block * bytesPerBlock + slot] = 0x00
                packed[block * bytesPerBlock + slot + 1] = 0x3C
            }
        }
        val input = FloatArray(inputDim) { rng.nextFloat() - 0.5f }
        val ref = FloatArray(outputDim)
        val got = FloatArray(outputDim)
        scalar(input, packed, inputDim, outputDim, ref)
        wasm(input, packed, inputDim, outputDim, got)
        if (exact) {
            assertClose(ref, got, what, rel = 1e-4f)
        } else {
            var err = 0.0; var sig = 0.0
            for (o in 0 until outputDim) { err += (ref[o] - got[o]).toDouble().let { it * it }; sig += ref[o].toDouble().let { it * it } }
            val relRms = sqrt(err / outputDim) / (sqrt(sig / outputDim) + 1e-9)
            assertTrue(relRms < 0.03, "$what: relRms=$relRms over $outputDim rows")
        }
    }

    @Test
    fun `K-quant and Q5 weights run on the sidecar and match the scalar kernels`() = runTest {
        WasmSimd.install()
        val p = checkNotNull(KernelRegistry.bestAvailable())
        assertEquals("wasm-simd128", p.name)
        // 37 rows: an odd count, so a threaded row split (none below 512) is not what is under test;
        // two or three blocks per row cover the block-outer accumulation and the per-row carry.
        packedParity("q4_k", 256, 144, intArrayOf(0, 2), 512, 37, exact = false,
            { i, w, id, od, o -> ScalarQ4_KMatmulKernel.matmul(i, 0, w, 0, id, od, o, 0) },
            { i, w, id, od, o -> checkNotNull(p.matmulQ4K()).matmul(i, 0, w, 0, id, od, o, 0) })
        packedParity("q6_k", 256, 210, intArrayOf(208), 512, 37, exact = false,
            { i, w, id, od, o -> ScalarQ6_KMatmulKernel.matmul(i, 0, w, 0, id, od, o, 0) },
            { i, w, id, od, o -> checkNotNull(p.matmulQ6K()).matmul(i, 0, w, 0, id, od, o, 0) })
        packedParity("q5_k", 256, 176, intArrayOf(0, 2), 512, 37, exact = true,
            { i, w, id, od, o -> ScalarQ5_KMatmulKernel.matmul(i, 0, w, 0, id, od, o, 0) },
            { i, w, id, od, o -> checkNotNull(p.matmulQ5K()).matmul(i, 0, w, 0, id, od, o, 0) })
        packedParity("q5_0", 32, 22, intArrayOf(0), 96, 37, exact = true,
            { i, w, id, od, o -> ScalarQ5_0MatmulKernel.matmul(i, 0, w, 0, id, od, o, 0) },
            { i, w, id, od, o -> checkNotNull(p.matmulQ5_0()).matmul(i, 0, w, 0, id, od, o, 0) })
        packedParity("q5_1", 32, 24, intArrayOf(0, 2), 96, 37, exact = true,
            { i, w, id, od, o -> ScalarQ5_1MatmulKernel.matmul(i, 0, w, 0, id, od, o, 0) },
            { i, w, id, od, o -> checkNotNull(p.matmulQ5_1()).matmul(i, 0, w, 0, id, od, o, 0) })
    }

    @Test
    fun `dense GEMM through the ops layer runs on the sidecar for every tile shape`() = runTest {
        val scalar = DirectCpuExecutionContext()
        WasmSimd.install()
        val fast = DirectCpuExecutionContext()
        assertIs<WasmCpuOps>(fast.ops)
        // Full 4×8 tiles, a 1-row remainder, a 6-column remainder and a k past one 256-block.
        for ((m, n, k) in listOf(Triple(9, 70, 300), Triple(1, 8, 5), Triple(4, 3, 1), Triple(5, 16, 256))) {
            val x = pseudo(m * k, m + n)
            val b = pseudo(k * n, k)
            val ref = scalar.ops.matmul(
                scalar.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, x),
                scalar.fromFloatArray<FP32, Float>(Shape(k, n), FP32::class, b),
            ).data.copyToFloatArray()
            val got = fast.ops.matmul(
                fast.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, x),
                fast.fromFloatArray<FP32, Float>(Shape(k, n), FP32::class, b),
            ).data.copyToFloatArray()
            assertClose(ref, got, "gemm $m×$k·$k×$n")
        }
    }

    @Test
    fun `row-major Q8_0 and Q4_0 weights match the scalar lane and run on the pack`() = runTest {
        val scalar = DirectCpuExecutionContext()
        val m = 5; val n = 70; val k = 96
        val x = pseudo(m * k, 3)
        val wq8 = q8(scalar, n, k, 7)
        val wq4 = q4(scalar, n, k, 9)
        val xs = scalar.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, x)
        val ref8 = scalar.ops.matmulWeightTransposed(xs, wq8).data.copyToFloatArray()
        val ref4 = scalar.ops.matmulWeightTransposed(xs, wq4).data.copyToFloatArray()

        WasmSimd.install()
        val fast = DirectCpuExecutionContext()
        val xf = fast.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, x)
        assertIs<Q8_0BlockTensorData>(wq8.data)
        val got8 = fast.ops.matmulWeightTransposed(xf, wq8).data.copyToFloatArray()
        val got4 = fast.ops.matmulWeightTransposed(xf, wq4).data.copyToFloatArray()
        assertClose(ref8, got8, "q8_0")
        assertClose(ref4, got4, "q4_0")
        // Second call hits the resident weight; same answer, bytes stay resident.
        assertClose(ref8, fast.ops.matmulWeightTransposed(xf, wq8).data.copyToFloatArray(), "q8_0 again")
        assertTrue(WasmKernels.residentWeightBytes() > 0)
        val names = KernelDispatch.kernels().map { it.name }
        assertTrue(names.any { it.startsWith("wasm-simd128") }, "pack registered: $names")
    }

    @Test
    fun `dense weight-transposed and dense GEMM match the scalar lane`() = runTest {
        val scalar = DirectCpuExecutionContext()
        val m = 6; val n = 37; val k = 45
        val x = pseudo(m * k, 11)
        val w = pseudo(n * k, 13)
        val b = pseudo(k * n, 17)
        val xs = scalar.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, x)
        val ws = scalar.fromFloatArray<FP32, Float>(Shape(n, k), FP32::class, w)
        val bs = scalar.fromFloatArray<FP32, Float>(Shape(k, n), FP32::class, b)
        val refWt = scalar.ops.matmulWeightTransposed(xs, ws).data.copyToFloatArray()
        val refMm = scalar.ops.matmul(xs, bs).data.copyToFloatArray()

        WasmSimd.install()
        val fast = DirectCpuExecutionContext()
        val xf = fast.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, x)
        val wf = fast.fromFloatArray<FP32, Float>(Shape(n, k), FP32::class, w)
        val bf = fast.fromFloatArray<FP32, Float>(Shape(k, n), FP32::class, b)
        assertClose(refWt, fast.ops.matmulWeightTransposed(xf, wf).data.copyToFloatArray(), "x·Wᵀ")
        val rank1 = fast.fromFloatArray<FP32, Float>(Shape(k), FP32::class, x.copyOfRange(0, k))
        assertClose(refWt.copyOfRange(0, n), fast.ops.matmulWeightTransposed(rank1, wf).data.copyToFloatArray(), "rank-1 x·Wᵀ")
        // The dense GEMM runs the sidecar through the WasmCpuOps override (heap operands) and through
        // KernelDispatch → Fp32ViewMatmulKernel(WasmFp32MatmulKernel) for view operands; exercise the
        // provider kernel directly as well.
        assertClose(refMm, fast.ops.matmul(xf, bf).data.copyToFloatArray(), "matmul")
        val direct = FloatArray(m * n)
        WasmFp32MatmulKernel.matmul(x, 0, k, b, 0, n, direct, 0, n, m, n, k)
        assertClose(refMm, direct, "provider gemm")
        // The direct `x · Wᵀ` provider kernel: whole-array weight (resident) and a windowed one (staged).
        val provider = checkNotNull(KernelRegistry.bestAvailable())
        val wt = checkNotNull(provider.matmulFp32Wt())
        assertTrue(provider.supports("matmul_wt", listOf("Float32", "Float32")))
        val directWt = FloatArray(m * n)
        wt.matmulWt(x, 0, w, 0, directWt, 0, m, n, k)
        assertClose(refWt, directWt, "provider x·Wᵀ")
        val window = FloatArray(1 + n * k + 3).also { w.copyInto(it, 1) }
        val windowed = FloatArray(2 + m * n)
        wt.matmulWt(x, 0, window, 1, windowed, 2, m, n, k)
        assertClose(refWt, windowed.copyOfRange(2, 2 + m * n), "windowed x·Wᵀ")
    }

    /**
     * The ternary kernels reach the sidecar through the same `KernelDispatch` keys the JVM FFM and
     * Android JNI packs serve, and agree with the Kotlin reference kernels. Row counts above the row
     * pool's threshold (512) so the mt sidecar splits the rows over its workers.
     */
    @Test
    fun `ternary weights run on the sidecar and match the reference kernels`() = runTest {
        WasmSimd.install()
        val rng = Random(31)

        // BITNET_B1_58, exact FP32 activations, two rows so the pack's row loop runs.
        run {
            val k = 256; val n = 600; val rows = 2
            val values = FloatArray(n * k) { (rng.nextInt(3) - 1) * 0.5f }
            val w = BitNetB158TensorData.fromFloats(Shape(n, k), values).packedView
            val a = TensorView.dense(Storage.Heap.wrap(FloatArray(rows * k) { rng.nextFloat() - 0.5f }), Shape(rows, k), FP32)
            val out = TensorView.dense(Storage.Heap.floats(rows * n), Shape(rows, n), FP32)
            val sink = RecordingTraceSink()
            KernelDispatch.matmul(a, w, out, Scope.Ambient, sink)
            assertEquals("ternary_f32_gemv/wasm-simd128", sink.eventsOf<TraceEvent.KernelRun>().single().kernel)
            val ref = TensorView.dense(Storage.Heap.floats(rows * n), Shape(rows, n), FP32)
            TernaryF32GemvKernel(TernaryF32GemvKernel.keyFor()).run(listOf(a, w), ref)
            assertClose(heapFloats(ref), heapFloats(out), "ternary f32 gemv", rel = 1e-4f)
        }

        // BITNET_PLANES: the fused four-plane lm_head, two calls per row, against the 8-plane reference.
        run {
            val k = 64; val n = 520
            val values = FloatArray(n * k) { (rng.nextFloat() - 0.5f) * 2f }
            val w = BitNetPlanesTensorData.fromFloats(Shape(n, k), values).packedView
            val a = TensorView.dense(Storage.Heap.wrap(FloatArray(k) { rng.nextFloat() - 0.5f }), Shape(1, k), FP32)
            val out = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
            val sink = RecordingTraceSink()
            KernelDispatch.matmul(a, w, out, Scope.Ambient, sink)
            assertEquals("ternary_planes_matmul/wasm-simd128", sink.eventsOf<TraceEvent.KernelRun>().single().kernel)
            val ref = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
            TernaryPlanesMatmulKernel(TernaryPlanesMatmulKernel.keyFor()).run(listOf(a, w), ref)
            assertClose(heapFloats(ref), heapFloats(out), "ternary planes lm_head", rel = 1e-4f)
        }

        // TQ2_0 with int8 activations: integer block sums, so the match is exact up to the scale products.
        run {
            val k = 768; val n = 600
            val values = FloatArray(n * k) { (rng.nextInt(3) - 1) * 0.5f }
            val w = TensorView.packed(
                Storage.Heap.wrap(TernaryCodec.encode(TensorEncoding.TQ2_0, values)), Shape(n, k),
                TensorEncoding.TQ2_0, TernaryBlockDecoder(TensorEncoding.TQ2_0),
            )
            val a = I8Absmax.requantize(
                TensorView.dense(Storage.Heap.wrap(FloatArray(k) { rng.nextFloat() - 0.5f }), Shape(1, k), FP32), Scope.Ambient,
            )
            val out = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
            val sink = RecordingTraceSink()
            KernelDispatch.matmul(a, w, out, Scope.Ambient, sink)
            assertEquals("bitnet_gemv/wasm-simd128", sink.eventsOf<TraceEvent.KernelRun>().single().kernel)
            val ref = TensorView.dense(Storage.Heap.floats(n), Shape(1, n), FP32)
            BitNetGemvKernel(BitNetGemvKernel.keyFor(w.format)).run(listOf(a, w), ref)
            assertClose(heapFloats(ref), heapFloats(out), "bitnet gemv tq2_0")
        }
        assertTrue(WasmKernels.residentWeightBytes() > 0, "the ternary weights stay resident")
    }

    /**
     * Resident activations: a matmul's output stays in the sidecar, the elementwise ops, activations
     * and reductions that follow read it in place and leave their result there too, and every value
     * agrees with the common scalar lane. Only the final read materialises a heap mirror.
     */
    @Test
    fun `an activation chain stays resident in the sidecar and matches the scalar lane`() = runTest {
        val scalar = DirectCpuExecutionContext()
        val m = 7; val n = 40; val k = 24
        val x = pseudo(m * k, 41)
        val w = pseudo(n * k, 43)
        val bias = pseudo(n, 47)
        val xs = scalar.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, x)
        val ws = scalar.fromFloatArray<FP32, Float>(Shape(n, k), FP32::class, w)
        val bs = scalar.fromFloatArray<FP32, Float>(Shape(n), FP32::class, bias)
        val two = scalar.fromFloatArray<FP32, Float>(Shape(1), FP32::class, floatArrayOf(2f))
        fun chain(ctx: DirectCpuExecutionContext, x: Tensor<FP32, Float>, w: Tensor<FP32, Float>, b: Tensor<FP32, Float>, c: Tensor<FP32, Float>, probe: (String, Tensor<FP32, Float>) -> Unit): Map<String, FloatArray> {
            val ops = ctx.ops
            val out = LinkedHashMap<String, FloatArray>()
            fun keep(name: String, t: Tensor<FP32, Float>) { probe(name, t); out[name] = t.data.copyToFloatArray() }
            val h = ops.matmulWeightTransposed(x, w); probe("h", h)
            val g = ops.gelu(h); probe("gelu", g)
            val ge = ops.gelu(h, approximate = false); keep("geluExact", ge)
            val s = ops.multiply(g, h); probe("mul", s)
            val o = ops.add(s, b); probe("addBias", o)
            keep("biasAdd", ops.add(b, s))
            keep("subScalar", ops.subtract(o, c))
            keep("scalarSub", ops.subtract(c, o))
            keep("div", ops.divide(o, ops.add(ops.relu(o), c)))
            keep("relu", ops.relu(o))
            keep("sigmoid", ops.sigmoid(o))
            keep("silu", ops.silu(o))
            keep("tanh", ops.tanh(o))
            keep("erf", ops.erf(o))
            keep("sumAll", ops.sum(o))
            keep("meanAll", ops.mean(o))
            keep("sumLast", ops.sum(o, -1))
            keep("meanFirst", ops.mean(o, 0))
            keep("out", o)
            return out
        }
        val ref = chain(scalar, xs, ws, bs, two) { _, _ -> }

        WasmSimd.install()
        val fast = DirectCpuExecutionContext()
        val xf = fast.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, x)
        val wf = fast.fromFloatArray<FP32, Float>(Shape(n, k), FP32::class, w)
        val bf = fast.fromFloatArray<FP32, Float>(Shape(n), FP32::class, bias)
        val cf = fast.fromFloatArray<FP32, Float>(Shape(1), FP32::class, floatArrayOf(2f))
        val before = WasmKernels.residentActivationBytes()
        val got = chain(fast, xf, wf, bf, cf) { name, t ->
            val d = t.data
            assertIs<WasmLinearTensorData<*>>(d, "$name should be resident")
            assertTrue(d.isResident, "$name alive")
            if (name == "h" || name == "gelu" || name == "mul" || name == "addBias") assertTrue(!d.isMirrored, "$name was not downloaded")
        }
        assertTrue(WasmKernels.residentActivationBytes() > before, "resident bytes grew")
        for ((name, values) in ref) {
            assertClose(values, checkNotNull(got[name]) { name }, name)
        }
        // Shapes of the reductions follow the common lane.
        assertEquals(m, got.getValue("sumLast").size)
        assertEquals(n, got.getValue("meanFirst").size)
        assertEquals(1, got.getValue("sumAll").size)
    }

    @Suppress("UNCHECKED_CAST")
    private fun lengthsTensor(ctx: DirectCpuExecutionContext, lengths: IntArray): Tensor<DType, *> =
        ctx.fromIntArray<Int32, Int>(Shape(lengths.size), Int32::class, lengths) as Tensor<DType, *>

    @Test
    fun `softmax, the normalisations and attention run fused in the sidecar and match the scalar lane`() = runTest {
        val scalar = DirectCpuExecutionContext()
        val rows = 16; val cols = 96
        val xArr = pseudo(rows * cols, 51, 3f)
        val gArr = FloatArray(cols) { 0.5f + (it % 7) * 0.1f }
        val bArr = FloatArray(cols) { ((it % 5) - 2) * 0.05f }
        val batch = 2; val heads = 4; val kvHeads = 2; val seqQ = 5; val seqKV = 9; val headDim = 16
        val qArr = pseudo(batch * heads * seqQ * headDim, 53)
        val kArr = pseudo(batch * kvHeads * seqKV * headDim, 55)
        val vArr = pseudo(batch * kvHeads * seqKV * headDim, 57)
        val lengths = intArrayOf(seqKV, 4)
        val maskArr = pseudo(seqQ * seqKV, 59)
        val wqArr = pseudo(cols * cols, 61)

        fun run(ctx: DirectCpuExecutionContext, probe: (String, Tensor<FP32, Float>) -> Unit): Map<String, FloatArray> {
            val ops = ctx.ops
            val out = LinkedHashMap<String, FloatArray>()
            fun keep(name: String, t: Tensor<FP32, Float>) { probe(name, t); out[name] = t.data.copyToFloatArray() }
            val x = ctx.fromFloatArray<FP32, Float>(Shape(rows, cols), FP32::class, xArr)
            keep("softmax", ops.softmax(x, -1))
            keep("logSoftmax", ops.logSoftmax(x, -1))
            keep("softmaxDim0", ops.softmax(x, 0))
            val gamma = ctx.fromFloatArray<FP32, Float>(Shape(cols), FP32::class, gArr)
            val beta = ctx.fromFloatArray<FP32, Float>(Shape(cols), FP32::class, bArr)
            val ln = LayerNormalization<FP32, Float>(intArrayOf(cols), eps = 1e-5, elementwiseAffine = true, name = "ln", initGamma = gamma, initBeta = beta)
            keep("layerNorm", ln.forward(x, ctx))
            val lnPlain = LayerNormalization<FP32, Float>(intArrayOf(cols), eps = 1e-5, elementwiseAffine = false, name = "lnPlain")
            keep("layerNormPlain", lnPlain.forward(x, ctx))
            (ops as? FusedNormOps)?.rmsNorm(x, gamma, 1e-6f, 1)?.let { keep("rmsNorm", it) }
            // On the sidecar the operands are made resident first (as a projection's output is), so
            // attention reads them in place: the probe checks they were never downloaded.
            val wasm = ops as? WasmCpuOps
            fun resident(t: Tensor<FP32, Float>) = wasm?.toResident(t) ?: t
            val q = resident(ctx.fromFloatArray<FP32, Float>(Shape(batch, heads, seqQ, headDim), FP32::class, qArr))
            val k = resident(ctx.fromFloatArray<FP32, Float>(Shape(batch, kvHeads, seqKV, headDim), FP32::class, kArr))
            val v = resident(ctx.fromFloatArray<FP32, Float>(Shape(batch, kvHeads, seqKV, headDim), FP32::class, vArr))
            keep("sdpa", ops.scaledDotProductAttention(q, k, v))
            probe("q", q); probe("k", k); probe("v", v)
            keep("sdpaCausal", ops.scaledDotProductAttention(q, k, v, causal = true))
            keep("sdpaLengths", ops.scaledDotProductAttention(q, k, v, causal = true, keyLengths = lengthsTensor(ctx, lengths)))
            keep("sdpaWindow", ops.scaledDotProductAttention(q, k, v, keyLengths = lengthsTensor(ctx, lengths), window = 3, rightContext = 1))
            val mask = ctx.fromFloatArray<FP32, Float>(Shape(seqQ, seqKV), FP32::class, maskArr)
            keep("sdpaMask", ops.scaledDotProductAttention(q, k, v, mask = mask, scale = 0.3f))
            val dense = ctx.fromFloatArray<FP32, Float>(Shape(batch, 1, seqQ, seqKV), FP32::class, AttentionMask.additive(batch, seqQ, seqKV, true, 3, 1, lengths))
            keep("sdpaDense", ops.scaledDotProductAttention(q, k, v, mask = dense))
            keep("sdpaStructured", ops.scaledDotProductAttention(q, k, v, causal = true, keyLengths = lengthsTensor(ctx, lengths), window = 3, rightContext = 1))
            keep("sdpaPadded", ops.scaledDotProductAttention(q, k, v, keyLengths = lengthsTensor(ctx, intArrayOf(0, seqKV))))
            // A projection's output goes straight into the normalisation without a download.
            val wq = ctx.fromFloatArray<FP32, Float>(Shape(cols, cols), FP32::class, wqArr)
            val proj = ops.matmulWeightTransposed(x, wq)
            keep("projNorm", lnPlain.forward(proj, ctx))
            probe("proj", proj)
            return out
        }
        val ref = run(scalar) { _, _ -> }
        assertTrue(ref.containsKey("rmsNorm").not() || scalar.ops !is FusedNormOps, "the scalar lane has no fused-norm hook")

        WasmSimd.install()
        val fast = DirectCpuExecutionContext()
        assertIs<FusedNormOps>(fast.ops)
        val got = run(fast) { name, t ->
            if (name == "softmaxDim0") return@run
            val d = t.data
            assertIs<WasmLinearTensorData<*>>(d, "$name should be resident")
            if (name == "proj" || name == "q" || name == "k" || name == "v") assertTrue(!d.isMirrored, "$name was not downloaded")
        }
        for ((name, values) in ref) {
            val (rel, abs) = if (name.startsWith("sdpa") || name.contains("Norm")) 1e-4f to 1e-5f else 1e-5f to 1e-6f
            val actual = checkNotNull(got[name]) { name }
            assertEquals(values.size, actual.size, "$name size")
            var maxAbs = 0f
            for (v in values) maxAbs = maxOf(maxAbs, kotlin.math.abs(v))
            val tol = maxAbs * rel + abs
            for (i in values.indices) assertTrue(kotlin.math.abs(values[i] - actual[i]) <= tol, "$name[$i]: ${actual[i]} vs ${values[i]} (tol $tol)")
        }
        assertTrue(got.containsKey("rmsNorm"), "rmsNorm served by the sidecar")
        // The structured rules equal the dense -1e30 mask on the sidecar kernel.
        val a = got.getValue("sdpaStructured"); val b = got.getValue("sdpaDense")
        for (i in a.indices) assertTrue(abs(a[i] - b[i]) <= 1e-6f, "structured vs dense [$i]")
        assertTrue(got.getValue("sdpaPadded").all { it.isFinite() }, "fully padded row is finite")
    }

    @Test
    fun `convolutions run as im2col and GEMM in the sidecar and match the scalar lane`() = runTest {
        val scalar = DirectCpuExecutionContext()
        // conv2d over a [2, 4, 9, 7] input: 3×3 same padding with and without bias, a 2×3 kernel with
        // stride, dilation and asymmetric padding, depthwise, two groups with stride, 1×1, the 4-D bias
        // form; conv1d over [2, 3, 17] with stride, dilation and groups; conv3d over [1, 2, 5, 6, 7].
        val x2Arr = pseudo(2 * 4 * 9 * 7, 71); val w2Arr = pseudo(6 * 4 * 3 * 3, 73); val b2Arr = pseudo(6, 75)
        val w2sArr = pseudo(5 * 4 * 2 * 3, 77)
        val wDwArr = pseudo(4 * 1 * 3 * 3, 79)
        val wGArr = pseudo(6 * 2 * 3 * 3, 81)
        val w11Arr = pseudo(8 * 4, 83)
        val x1Arr = pseudo(2 * 3 * 17, 85); val w1Arr = pseudo(4 * 3 * 5, 87); val b1Arr = pseudo(4, 89)
        val w1gArr = pseudo(6 * 1 * 3, 91)
        val x3Arr = pseudo(1 * 2 * 5 * 6 * 7, 93); val w3Arr = pseudo(3 * 2 * 2 * 3 * 3, 95); val b3Arr = pseudo(3, 97)

        fun run(ctx: DirectCpuExecutionContext, probe: (String, Tensor<FP32, Float>) -> Unit): Map<String, FloatArray> {
            val ops = ctx.ops
            val out = LinkedHashMap<String, FloatArray>()
            fun keep(name: String, t: Tensor<FP32, Float>) { probe(name, t); out[name] = t.data.copyToFloatArray() }
            val x2 = ctx.fromFloatArray<FP32, Float>(Shape(2, 4, 9, 7), FP32::class, x2Arr)
            val w2 = ctx.fromFloatArray<FP32, Float>(Shape(6, 4, 3, 3), FP32::class, w2Arr)
            val b2 = ctx.fromFloatArray<FP32, Float>(Shape(6), FP32::class, b2Arr)
            keep("conv2d", ops.conv2d(x2, w2, b2, 1 to 1, 1 to 1, 1 to 1, 1))
            keep("conv2dNoBias", ops.conv2d(x2, w2, null, 1 to 1, 1 to 1, 1 to 1, 1))
            keep("conv2dBias4d", ops.conv2d(x2, w2, ctx.fromFloatArray(Shape(1, 6, 1, 1), FP32::class, b2Arr), 1 to 1, 1 to 1, 1 to 1, 1))
            val w2s = ctx.fromFloatArray<FP32, Float>(Shape(5, 4, 2, 3), FP32::class, w2sArr)
            keep("conv2dStrideDilation", ops.conv2d(x2, w2s, null, 2 to 1, 0 to 2, 2 to 1, 1))
            val wDw = ctx.fromFloatArray<FP32, Float>(Shape(4, 1, 3, 3), FP32::class, wDwArr)
            keep("conv2dDepthwise", ops.conv2d(x2, wDw, null, 1 to 1, 1 to 1, 1 to 1, 4))
            val wG = ctx.fromFloatArray<FP32, Float>(Shape(6, 2, 3, 3), FP32::class, wGArr)
            keep("conv2dGrouped", ops.conv2d(x2, wG, b2, 1 to 2, 1 to 1, 1 to 1, 2))
            val w11 = ctx.fromFloatArray<FP32, Float>(Shape(8, 4, 1, 1), FP32::class, w11Arr)
            keep("conv2d1x1", ops.conv2d(x2, w11, null, 1 to 1, 0 to 0, 1 to 1, 1))
            val x1 = ctx.fromFloatArray<FP32, Float>(Shape(2, 3, 17), FP32::class, x1Arr)
            val w1 = ctx.fromFloatArray<FP32, Float>(Shape(4, 3, 5), FP32::class, w1Arr)
            val b1 = ctx.fromFloatArray<FP32, Float>(Shape(4), FP32::class, b1Arr)
            keep("conv1d", ops.conv1d(x1, w1, b1, 2, 2, 1, 1))
            keep("conv1dDilation", ops.conv1d(x1, w1, null, 1, 4, 2, 1))
            val w1g = ctx.fromFloatArray<FP32, Float>(Shape(6, 1, 3), FP32::class, w1gArr)
            keep("conv1dGrouped", ops.conv1d(x1, w1g, null, 1, 1, 1, 3))
            val x3 = ctx.fromFloatArray<FP32, Float>(Shape(1, 2, 5, 6, 7), FP32::class, x3Arr)
            val w3 = ctx.fromFloatArray<FP32, Float>(Shape(3, 2, 2, 3, 3), FP32::class, w3Arr)
            val b3 = ctx.fromFloatArray<FP32, Float>(Shape(3), FP32::class, b3Arr)
            keep("conv3d", ops.conv3d(x3, w3, b3, Triple(1, 2, 1), Triple(1, 1, 0), Triple(1, 1, 1), 1))
            // A layer's chain: the convolution's output feeds the activation without a download, and
            // a resident input (the previous layer's output) is read in place.
            keep("convRelu", ops.relu(ops.conv2d(x2, w2, b2, 1 to 1, 1 to 1, 1 to 1, 1)))
            val wasm = ops as? WasmCpuOps
            val xResident = wasm?.toResident(x2) ?: x2
            keep("conv2dResidentInput", ops.conv2d(xResident, w2, b2, 1 to 1, 1 to 1, 1 to 1, 1))
            probe("xResident", xResident)
            return out
        }
        val ref = run(scalar) { _, _ -> }

        WasmSimd.install()
        val fast = DirectCpuExecutionContext()
        assertIs<WasmCpuOps>(fast.ops)
        val got = run(fast) { name, t ->
            val d = t.data
            assertIs<WasmLinearTensorData<*>>(d, "$name should be resident")
            if (name == "xResident") assertTrue(!d.isMirrored, "the resident input was not downloaded")
        }
        for ((name, values) in ref) assertClose(values, checkNotNull(got[name]) { name }, name)

        // A weight is a parameter: resident after its first call, not re-uploaded by the next one.
        val x2 = fast.fromFloatArray<FP32, Float>(Shape(2, 4, 9, 7), FP32::class, x2Arr)
        val w2 = fast.fromFloatArray<FP32, Float>(Shape(6, 4, 3, 3), FP32::class, w2Arr)
        val before = WasmKernels.residentWeightBytes()
        fast.ops.conv2d(x2, w2, null, 1 to 1, 1 to 1, 1 to 1, 1)
        val held = WasmKernels.residentWeightBytes()
        assertEquals(before + w2Arr.size * 4L, held, "the weight went resident on its first call")
        fast.ops.conv2d(x2, w2, null, 1 to 1, 1 to 1, 1 to 1, 1)
        assertEquals(held, WasmKernels.residentWeightBytes(), "a second call with the same weight uploads nothing")
    }

    @Test
    fun `heap operands keep the common path while transcendental activations and matmul outputs go resident`() = runTest {
        WasmSimd.install()
        val ctx = DirectCpuExecutionContext()
        assertIs<WasmCpuOps>(ctx.ops)
        val a = ctx.fromFloatArray<FP32, Float>(Shape(3, 5), FP32::class, pseudo(15, 1))
        val b = ctx.fromFloatArray<FP32, Float>(Shape(3, 5), FP32::class, pseudo(15, 2))
        // Two heap operands: no upload, the common scalar lane answers on the heap.
        assertIs<DenseFloatArrayTensorData<*>>(ctx.ops.add(a, b).data)
        assertIs<DenseFloatArrayTensorData<*>>(ctx.ops.relu(a).data)
        assertIs<DenseFloatArrayTensorData<*>>(ctx.ops.sum(a).data)
        // A transcendental on a heap operand is worth the upload; its output is resident.
        val g = ctx.ops.gelu(a)
        assertIs<WasmLinearTensorData<*>>(g.data)
        // ... and from there the chain is resident, including a heap partner.
        assertIs<WasmLinearTensorData<*>>(ctx.ops.add(g, b).data)
        assertIs<WasmLinearTensorData<*>>(ctx.ops.relu(g).data)
        assertIs<WasmLinearTensorData<*>>(ctx.ops.sum(g).data)
        // Explicit upload once.
        val r = (ctx.ops as WasmCpuOps).toResident(a)
        assertIs<WasmLinearTensorData<*>>(r.data)
        assertClose(a.data.copyToFloatArray(), r.data.copyToFloatArray(), "toResident")
        assertIs<WasmLinearTensorData<*>>(ctx.ops.add(r, b).data)
        // A resident tensor reads like a heap one: elements, the view, and a common op that has no override.
        val gd = g.data as WasmLinearTensorData<*>
        val expected = DirectCpuExecutionContext().let { s -> s.ops.gelu(s.fromFloatArray<FP32, Float>(Shape(3, 5), FP32::class, pseudo(15, 1))).data.copyToFloatArray() }
        assertClose(expected, g.data.copyToFloatArray(), "gelu via mirror")
        assertEquals(expected[7], gd.get(1, 2))
        assertTrue(gd.isMirrored)
        assertClose(expected, ctx.ops.transpose(ctx.ops.transpose(g)).data.copyToFloatArray(), "common op on a resident tensor")
    }

    @Test
    fun `a resident tensor's view is the sidecar bytes and a slice of it stays resident`() = runTest {
        WasmSimd.install()
        val ctx = DirectCpuExecutionContext()
        val values = pseudo(15, 1)
        val a = ctx.fromFloatArray<FP32, Float>(Shape(3, 5), FP32::class, values)
        val expected = DirectCpuExecutionContext().let { s -> s.ops.gelu(s.fromFloatArray<FP32, Float>(Shape(3, 5), FP32::class, values)).data.copyToFloatArray() }
        val g = ctx.ops.gelu(a)
        val gd = assertIs<WasmLinearTensorData<*>>(g.data)

        // The view says where the bytes are (SKEEP-003, #1507): dense FP32 over the sidecar
        // storage. Asking for it, and reading an element through it, downloads nothing.
        val view = gd.view
        assertIs<WasmLinearStorage>(view.storage)
        assertTrue(view.isContiguous)
        assertEquals(Shape(3, 5), view.shape)
        assertEquals(expected[7], view.get(1, 2), 1e-5f)
        assertTrue(!gd.isMirrored, "the view is not the heap mirror")

        // A contiguous slice (row 1) is found in the sidecar by its view: a sidecar op reads it in
        // place and answers resident, still without a download of the source.
        val row = g.sliceView { segment { at(1) }; segment { all() } }
        val relu = ctx.ops.relu(row)
        assertIs<WasmLinearTensorData<*>>(relu.data, "relu of a resident slice runs in the sidecar")
        assertTrue(!gd.isMirrored, "the slice was read in place")
        val rowExpected = FloatArray(5) { maxOf(expected[5 + it], 0f) }
        assertClose(rowExpected, relu.data.copyToFloatArray(), "relu of a resident slice")

        // A common op without an override reads the slice through the heap mirror.
        val t = ctx.ops.transpose(g.sliceView { segment { range(1, 3) }; segment { all() } })
        val tExpected = FloatArray(10) { expected[5 + (it % 2) * 5 + it / 2] }
        assertClose(tExpected, t.data.copyToFloatArray(), "common op on a resident slice")
        assertTrue(gd.isMirrored)
    }

    /**
     * Where a weight's sidecar copy comes from is decided by what the data says, not by its class
     * (SKEEP-003, #1507): a resident weight is read in place, a heap array of its own is uploaded
     * once, a window of a larger array and anything a forward scope owns is staged per call.
     */
    @Test
    fun `a weight is read in place, uploaded once or staged by what its data says`() = runTest {
        val scalar = DirectCpuExecutionContext()
        val m = 3; val n = 7; val k = 9
        val x = pseudo(m * k, 21)
        val w = pseudo(n * k, 23)
        val ref = scalar.ops.matmulWeightTransposed(
            scalar.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, x),
            scalar.fromFloatArray<FP32, Float>(Shape(n, k), FP32::class, w),
        ).data.copyToFloatArray()

        WasmSimd.install()
        val ctx = DirectCpuExecutionContext()
        val ops = assertIs<WasmCpuOps>(ctx.ops)
        val xf = ctx.fromFloatArray<FP32, Float>(Shape(m, k), FP32::class, x)

        // Resident: no heap mirror is downloaded and nothing is uploaded a second time.
        val resident = ops.toResident(ctx.fromFloatArray<FP32, Float>(Shape(n, k), FP32::class, w.copyOf()))
        val rd = assertIs<WasmLinearTensorData<*>>(resident.data)
        var held = WasmKernels.residentWeightBytes()
        assertClose(ref, ops.matmulWeightTransposed(xf, resident).data.copyToFloatArray(), "resident weight")
        assertTrue(!rd.isMirrored, "a resident weight is read where it is")
        assertEquals(held, WasmKernels.residentWeightBytes(), "and not uploaded again")

        // A heap array of its own: uploaded once, kept.
        val own = w.copyOf()
        val heapWeight = ctx.fromFloatArray<FP32, Float>(Shape(n, k), FP32::class, own)
        assertClose(ref, ops.matmulWeightTransposed(xf, heapWeight).data.copyToFloatArray(), "heap weight")
        assertEquals(held + n * k * 4, WasmKernels.residentWeightBytes(), "uploaded")
        assertClose(ref, ops.matmulWeightTransposed(xf, heapWeight).data.copyToFloatArray(), "heap weight again")
        assertEquals(held + n * k * 4, WasmKernels.residentWeightBytes(), "once")
        held = WasmKernels.residentWeightBytes()

        // A window of a larger array, for both operands: staged from its offset, nothing kept.
        val slab = FloatArray(4 + n * k + m * k + 5) { 999f }
        w.copyInto(slab, 4)
        x.copyInto(slab, 4 + n * k)
        val heap = Storage.Heap.wrap(slab)
        @Suppress("UNCHECKED_CAST")
        val windowW = ctx.fromData(
            StorageFloatTensorData<FP32>(Shape(n, k), heap.slice(4L * 4, n * k * 4L)) as TensorData<FP32, Float>, FP32::class,
        )
        @Suppress("UNCHECKED_CAST")
        val windowX = ctx.fromData(
            StorageFloatTensorData<FP32>(Shape(m, k), heap.slice((4L + n * k) * 4, m * k * 4L)) as TensorData<FP32, Float>, FP32::class,
        )
        assertClose(ref, ops.matmulWeightTransposed(windowX, windowW).data.copyToFloatArray(), "windowed operands")
        assertEquals(held, WasmKernels.residentWeightBytes(), "a window is staged, not kept")

        // A whole array that a forward scope owns: staged, because the scope reuses the array.
        val scoped = Storage.Heap.floats(n * k, sk.ainet.lang.memory.ScopeKind.FORWARD)
        w.copyInto(checkNotNull(scoped.floats))
        @Suppress("UNCHECKED_CAST")
        val scopedW = ctx.fromData(StorageFloatTensorData<FP32>(Shape(n, k), scoped) as TensorData<FP32, Float>, FP32::class)
        assertClose(ref, ops.matmulWeightTransposed(xf, scopedW).data.copyToFloatArray(), "forward-scoped weight")
        assertEquals(held, WasmKernels.residentWeightBytes(), "a forward-scoped array is not kept")
        // ... so the next step's values in the same array are the ones that are read.
        for (i in 0 until n * k) scoped.floats!![i] = 2f * w[i]
        assertClose(FloatArray(ref.size) { 2f * ref[it] }, ops.matmulWeightTransposed(xf, scopedW).data.copyToFloatArray(), "after the array was reused")
        WasmKernels.release(own)
    }

    @Test
    fun `scoped resident outputs are freed at reset and the budget falls back to the heap`() = runTest {
        WasmSimd.install()
        assertTrue(StorageCapabilities.current().supportsOffHeap, "the hook binds HOST_OFFHEAP while installed")
        val scope = ForwardScope(slabFloats = 0)
        val ctx = ScopedExecutionContext(DirectCpuExecutionContext(), scope)
        val x = ctx.fromFloatArray<FP32, Float>(Shape(4, 8), FP32::class, pseudo(32, 9))
        val before = WasmKernels.residentActivationBytes()
        val y = ctx.ops.gelu(x)
        val yd = assertIs<WasmLinearTensorData<*>>(y.data)
        assertEquals(sk.ainet.lang.memory.ScopeKind.FORWARD, yd.storage.scope)
        assertEquals(before + 32 * 4, WasmKernels.residentActivationBytes())
        val values = y.data.copyToFloatArray()
        scope.reset()
        assertEquals(before, WasmKernels.residentActivationBytes(), "reset freed the sidecar bytes")
        assertTrue(!yd.isResident)
        assertFailsWith<StorageClosedException> { y.data.copyToFloatArray() }
        assertEquals(32, values.size)

        // Budget: the kernel still runs in the sidecar, the result comes back to the heap.
        val saved = WasmSimd.residentBudgetBytes
        try {
            WasmSimd.residentBudgetBytes = 0L
            val plain = DirectCpuExecutionContext()
            val h = plain.fromFloatArray<FP32, Float>(Shape(4, 8), FP32::class, pseudo(32, 9))
            val z = plain.ops.gelu(h)
            assertIs<DenseFloatArrayTensorData<*>>(z.data)
            assertClose(values, z.data.copyToFloatArray(), "budget fallback")
        } finally {
            WasmSimd.residentBudgetBytes = saved
        }

        // The storage itself: zeroed allocation, byte round trip, slice, close.
        val s = assertIs<WasmLinearStorage>(PlatformStorage.allocate(64, MemoryDomain.HOST_OFFHEAP))
        val zeros = ByteArray(64) { 1 }
        s.copyInto(zeros)
        assertTrue(zeros.all { it == 0.toByte() }, "allocated zeroed")
        val bytes = ByteArray(64) { it.toByte() }
        s.copyFrom(bytes)
        val back = ByteArray(16)
        s.slice(8, 16).copyInto(back)
        assertTrue(back.contentEquals(bytes.copyOfRange(8, 24)), "slice reads the sub-range")
        val held = WasmKernels.residentActivationBytes()
        s.close()
        assertEquals(held - 64, WasmKernels.residentActivationBytes())
        WasmSimd.uninstall()
        assertTrue(!StorageCapabilities.current().supportsOffHeap, "heap only once uninstalled")
    }
}
