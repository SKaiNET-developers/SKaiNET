package sk.ainet.exec.golden

import sk.ainet.lang.tensor.ops.turboquant.TurboQuantCodec
import sk.ainet.lang.tensor.ops.turboquant.TurboQuantConfig
import kotlin.test.Test

/**
 * SKEEP-003 golden gate, TurboQuant: encoded bytes, scales and the decoded vector must stay
 * bit-identical for the same input and seed (the KV-cache compression path).
 */
class TurboQuantGoldenTest {

    private fun golden(tag: String, config: TurboQuantConfig) {
        val input = GoldenSupport.floats(128, 0x5EED_0003L, scale = 3f)
        val block = TurboQuantCodec.encode(input, config)
        GoldenSupport.check("turboquant/$tag/codes", GoldenSupport.digest(block.packedCodes))
        GoldenSupport.check("turboquant/$tag/scales", GoldenSupport.digest(block.scales))
        GoldenSupport.check("turboquant/$tag/decoded", GoldenSupport.digest(TurboQuantCodec.decode(block)))
    }

    @Test fun polar4() = golden("polar4", TurboQuantConfig.polarOnly(bits = 4, seed = 7))
    @Test fun polar3() = golden("polar3", TurboQuantConfig.polarOnly(bits = 3, seed = 7))
    @Test fun polar4Qjl1() = golden("polar4-qjl1", TurboQuantConfig.polarPlusQjl(bits = 4, residualBits = 1, seed = 7))
}
