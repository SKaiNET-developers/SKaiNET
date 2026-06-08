package sk.ainet.compile.minerva

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import sk.ainet.compile.export.GraphExportArtifactRole
import sk.ainet.compile.export.GraphExportContext

class MinervaNpzModelWriterTest {

    @Test
    fun writesDeterministicNpzSchemaForTwoLayerMlp() {
        val context = minervaContext(projectName = "TwoLayerMlp")
        val intermediate = MinervaGraphCanonicalizer().convert(twoLayerMinervaMlpGraph(), context)
        val writer = MinervaNpzModelWriter()

        val first = writer.write(intermediate, context)
        val second = writer.write(
            MinervaGraphCanonicalizer().convert(twoLayerMinervaMlpGraph(), minervaContext(projectName = "TwoLayerMlp")),
            minervaContext(projectName = "SecondWriterContext")
        )

        assertTrue(first.bytes.contentEquals(second.bytes))
        assertEquals(
            listOf(
                "schema_version",
                "layer_count",
                "input_shape",
                "output_shape",
                "layer_0_w",
                "layer_0_b",
                "layer_0_act",
                "layer_0_input_shape",
                "layer_0_output_shape",
                "layer_1_w",
                "layer_1_b",
                "layer_1_act",
                "layer_1_input_shape",
                "layer_1_output_shape"
            ),
            first.arrayNames
        )
        assertEquals(1, first.schemaVersion)
        assertEquals("1", first.metadata["schemaVersion"])
        assertEquals("2", first.metadata["layerCount"])
        assertEquals("1x4", first.metadata["inputShape"])
        assertEquals("1x2", first.metadata["outputShape"])
        assertEquals(listOf(4, 3), first.array("layer_0_w").shape)
        assertEquals(12, first.array("layer_0_w").floatData.size)
        assertEquals(listOf("relu"), first.array("layer_0_act").stringData)
        assertEquals(listOf("sigmoid"), first.array("layer_1_act").stringData)
        assertTrue(context.artifacts.any { it.path == "model.npz" && it.role == GraphExportArtifactRole.INTERMEDIATE })
        assertTrue(context.diagnostics.any { it.code == "minerva.npz.completed" })
    }

    @Test
    fun generatedArchiveContainsReadableNpyEntries() {
        val context = minervaContext(projectName = "ReadableNpz")
        val intermediate = MinervaGraphCanonicalizer().convert(twoLayerMinervaMlpGraph(), context)
        val model = MinervaNpzModelWriter().write(intermediate, context)

        val entries = readZipStoreEntries(model.bytes)

        assertEquals(model.arrayNames.map { "$it.npy" }, entries.map { it.name })
        assertTrue(entries.all { it.data.startsWithNpyMagic() })
        assertTrue(entries.single { it.name == "layer_0_w.npy" }.npyHeader().contains("'descr': '<f4'"))
        assertTrue(entries.single { it.name == "layer_0_w.npy" }.npyHeader().contains("'shape': (4, 3)"))
        assertTrue(entries.single { it.name == "layer_1_act.npy" }.npyHeader().contains("'descr': '<U7'"))
        assertTrue(entries.single { it.name == "layer_1_act.npy" }.npyHeader().contains("'shape': ()"))
    }

    @Test
    fun missingWeightValuesProduceTypedSchemaError() {
        val context = minervaContext(projectName = "BrokenNpz")
        val intermediate = MinervaGraphCanonicalizer().convert(validMinervaMlpGraph(), context)
        val layer = intermediate.layers.single()
        val broken = intermediate.copy(
            layers = listOf(layer.copy(weights = layer.weights.copy(values = null))),
            tensors = intermediate.tensors.map { tensor ->
                if (tensor.id == layer.weights.id) tensor.copy(values = null) else tensor
            }
        )

        val exception = assertFailsWith<MinervaNpzSchemaException> {
            MinervaNpzModelWriter().write(broken, context)
        }

        assertEquals("minerva.npz.missing_values", exception.code)
        assertEquals(layer.id, exception.layerId)
        assertEquals("layer_0_w", exception.arrayName)
        assertEquals(layer.weights.id, exception.details["tensorId"])
    }

    private fun minervaContext(projectName: String): GraphExportContext {
        val options = minervaTestOptions(projectName = projectName)
        return GraphExportContext(
            backendName = MinervaExportBackend.backendName,
            targetName = options.projectName,
            metadata = options.toMetadata()
        )
    }

    private fun MinervaNpzModel.array(name: String): MinervaNpzArray {
        return arrays.single { it.name == name }
    }

    private fun ByteArray.startsWithNpyMagic(): Boolean {
        return size >= 6 &&
            this[0] == 0x93.toByte() &&
            this[1] == 'N'.code.toByte() &&
            this[2] == 'U'.code.toByte() &&
            this[3] == 'M'.code.toByte() &&
            this[4] == 'P'.code.toByte() &&
            this[5] == 'Y'.code.toByte()
    }

    private fun ZipStoreEntry.npyHeader(): String {
        val headerLength = data.readShortLE(offset = 8)
        return data.copyOfRange(10, 10 + headerLength).decodeToString()
    }

    private fun readZipStoreEntries(bytes: ByteArray): List<ZipStoreEntry> {
        val entries = mutableListOf<ZipStoreEntry>()
        var offset = 0
        while (offset + LOCAL_HEADER_SIZE <= bytes.size && bytes.readIntLE(offset) == LOCAL_FILE_HEADER_SIGNATURE) {
            val compressedSize = bytes.readIntLE(offset + 18)
            val nameLength = bytes.readShortLE(offset + 26)
            val extraLength = bytes.readShortLE(offset + 28)
            val nameStart = offset + LOCAL_HEADER_SIZE
            val dataStart = nameStart + nameLength + extraLength
            val dataEnd = dataStart + compressedSize
            val name = bytes.copyOfRange(nameStart, nameStart + nameLength).decodeToString()
            entries += ZipStoreEntry(name = name, data = bytes.copyOfRange(dataStart, dataEnd))
            offset = dataEnd
        }
        assertTrue(entries.isNotEmpty())
        assertEquals(CENTRAL_DIRECTORY_SIGNATURE, bytes.readIntLE(offset))
        return entries
    }

    private fun ByteArray.readShortLE(offset: Int): Int {
        return (this[offset].toInt() and 0xff) or
            ((this[offset + 1].toInt() and 0xff) shl 8)
    }

    private fun ByteArray.readIntLE(offset: Int): Int {
        return readShortLE(offset) or (readShortLE(offset + 2) shl 16)
    }

    private data class ZipStoreEntry(val name: String, val data: ByteArray)

    private companion object {
        const val LOCAL_FILE_HEADER_SIGNATURE: Int = 0x04034b50
        const val CENTRAL_DIRECTORY_SIGNATURE: Int = 0x02014b50
        const val LOCAL_HEADER_SIZE: Int = 30
    }
}
