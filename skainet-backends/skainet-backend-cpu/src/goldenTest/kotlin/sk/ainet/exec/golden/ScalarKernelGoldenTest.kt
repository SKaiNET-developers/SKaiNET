package sk.ainet.exec.golden

import sk.ainet.exec.golden.GoldenSupport.Packed
import sk.ainet.exec.kernel.ScalarKernelProvider
import kotlin.test.Test
import kotlin.test.assertNotNull

/**
 * SKEEP-003 golden gate, kernel half: the pure-Kotlin scalar packed matmul kernels (the reference
 * tier every SIMD/native kernel is validated against) must stay bit-identical for the same bytes
 * and activations. Guards the KernelKey / registry migration (M1) — a re-routed dispatch that
 * still lands on these kernels produces these exact outputs.
 */
class ScalarKernelGoldenTest {

    private companion object {
        const val OUT = 4
        const val BLOCKS = 3
        const val BATCH = 3
        const val SEED = 0x5EED_0002L
    }

    private fun golden(
        p: Packed,
        run: (input: FloatArray, inOff: Int, weight: ByteArray, inputDim: Int, outputDim: Int, out: FloatArray, outOff: Int) -> Unit,
    ) {
        val inputDim = BLOCKS * p.blockSize
        val blocks = GoldenSupport.weightBlocks(p, OUT, BLOCKS, SEED)
        val weight = GoldenSupport.blockMajor(blocks)
        val input = GoldenSupport.floats(BATCH * inputDim, SEED + 100)
        val out = FloatArray(BATCH * OUT)
        for (r in 0 until BATCH) run(input, r * inputDim, weight, inputDim, OUT, out, r * OUT)
        GoldenSupport.check("scalar-matmul/${p.name}", GoldenSupport.digest(out))
    }

    @Test fun q4_0() { val k = assertNotNull(ScalarKernelProvider.matmulQ4_0()); golden(Packed.Q4_0) { i, io, w, n, m, o, oo -> k.matmul(i, io, w, 0, n, m, o, oo) } }
    @Test fun q5_0() { val k = assertNotNull(ScalarKernelProvider.matmulQ5_0()); golden(Packed.Q5_0) { i, io, w, n, m, o, oo -> k.matmul(i, io, w, 0, n, m, o, oo) } }
    @Test fun q5_1() { val k = assertNotNull(ScalarKernelProvider.matmulQ5_1()); golden(Packed.Q5_1) { i, io, w, n, m, o, oo -> k.matmul(i, io, w, 0, n, m, o, oo) } }
    @Test fun q8_0() { val k = assertNotNull(ScalarKernelProvider.matmulQ8_0()); golden(Packed.Q8_0) { i, io, w, n, m, o, oo -> k.matmul(i, io, w, 0, n, m, o, oo) } }
    @Test fun q4_K() { val k = assertNotNull(ScalarKernelProvider.matmulQ4K()); golden(Packed.Q4_K) { i, io, w, n, m, o, oo -> k.matmul(i, io, w, 0, n, m, o, oo) } }
    @Test fun q5_K() { val k = assertNotNull(ScalarKernelProvider.matmulQ5K()); golden(Packed.Q5_K) { i, io, w, n, m, o, oo -> k.matmul(i, io, w, 0, n, m, o, oo) } }
    @Test fun q6_K() { val k = assertNotNull(ScalarKernelProvider.matmulQ6K()); golden(Packed.Q6_K) { i, io, w, n, m, o, oo -> k.matmul(i, io, w, 0, n, m, o, oo) } }
}
