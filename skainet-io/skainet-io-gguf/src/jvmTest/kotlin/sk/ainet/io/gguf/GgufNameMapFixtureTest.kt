package sk.ainet.io.gguf

import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.weights.TransformerNameMaps
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * SKEEP-003 M0-A2 on real files: every tensor of a reference GGUF maps to a TensorId with zero
 * unmapped names. Fixture-gated like the tokenizer fixture tests: looks for the reference GGUFs in
 * `-Dskainet.test.fixturesDir` (default `build/test-fixtures` of skainet-io-core, where
 * `downloadQwenTokenizerFixtures` puts `Qwen2.5-0.5B-Instruct-Q8_0.gguf`) and prints `[skip]`
 * when a file is absent. Llama-3.2-1B and Gemma-3-1B are gated models — drop their GGUFs into the
 * fixtures dir to run those cases.
 */
class GgufNameMapFixtureTest {

    private val fixturesDir: File = File(
        System.getProperty("skainet.test.fixturesDir") ?: "../skainet-io-core/build/test-fixtures",
    )

    private fun check(fileName: String, expectedArchitecture: String) {
        val f = File(fixturesDir, fileName)
        if (!f.isFile) { println("[skip] $fileName not present in $fixturesDir"); return }
        JvmRandomAccessSource.open(f).use { src ->
            val reader = StreamingGGUFReader.open(src)
            assertEquals(expectedArchitecture, reader.fields["general.architecture"])
            val map = assertNotNull(reader.nameMap(), "name map for $expectedArchitecture")
            val names = reader.tensors.map { it.name }
            assertEquals(emptyList(), map.unmapped(names), "$fileName: unmapped tensor names")
            val ids = reader.tensorIds()
            assertEquals(names.size, ids.values.filterNotNull().toSet().size, "$fileName: distinct ids")
            assertTrue(names.isNotEmpty())
            println("[ok] $fileName: ${names.size} tensors → TensorIds (${map.family}/${map.format})")
        }
    }

    @Test fun qwen25_05b() = check("Qwen2.5-0.5B-Instruct-Q8_0.gguf", "qwen2")
    @Test fun llama32_1b() = check("Llama-3.2-1B-Instruct-Q4_K_M.gguf", "llama")
    @Test fun gemma3_1b() = check("gemma-3-1b-it-Q4_K_M.gguf", "gemma3")
}
