package sk.ainet.compile.minerva

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import sk.ainet.compile.export.GraphExportContext
import sk.ainet.compile.export.GraphExportStage

class MinervaGraphCanonicalizerTest {

    @Test
    fun lowersSupportedMlpPatternIntoDenseLayerIr() {
        val options = minervaTestOptions(projectName = "TinyMlp")
        val context = GraphExportContext(
            backendName = MinervaExportBackend.backendName,
            targetName = options.projectName,
            metadata = options.toMetadata()
        )

        val intermediate = MinervaGraphCanonicalizer().convert(validMinervaMlpGraph(), context)
        val layer = intermediate.layers.single()

        assertEquals("TinyMlp", intermediate.projectName)
        assertEquals(MinervaTarget.ATMEGA328P, intermediate.target)
        assertEquals(MinervaQuantization.Q8, intermediate.quantization)
        assertEquals(1, intermediate.layerCount)
        assertEquals(MinervaLayerKind.DENSE, layer.kind)
        assertEquals(MinervaTensorRole.INPUT, layer.input.role)
        assertEquals(MinervaTensorRole.WEIGHT, layer.weights.role)
        assertEquals(MinervaTensorRole.BIAS, layer.bias?.role)
        assertEquals(MinervaTensorRole.OUTPUT, layer.output.role)
        assertEquals(MinervaActivation.RELU, layer.activation)
        assertEquals(listOf("matmul", "bias_add", "relu"), layer.sourceNodeIds)
        assertEquals(listOf(1, 3), layer.output.shape)
        assertEquals(layer.weights.elementCount, layer.weights.values?.size)
        assertEquals(layer.bias?.elementCount, layer.bias?.values?.size)
        assertTrue(layer.hasBias)
        assertTrue(context.diagnostics.any { it.code == "minerva.lowering.started" })
        assertTrue(context.diagnostics.any { it.code == "minerva.lowering.completed" })
    }

    @Test
    fun loweredIntermediateCollectsStableTensorRefs() {
        val context = GraphExportContext(
            backendName = MinervaExportBackend.backendName,
            targetName = "TinyMlp",
            metadata = minervaTestOptions().toMetadata()
        )

        val intermediate = MinervaGraphCanonicalizer().convert(validMinervaMlpGraph(), context)

        assertEquals(intermediate.input, intermediate.layers.first().input)
        assertEquals(intermediate.output, intermediate.layers.last().output)
        assertTrue(intermediate.tensors.any { it.id == "input_input_x" })
        assertTrue(intermediate.tensors.any { it.id == "weight_weight_w" })
        assertTrue(intermediate.tensors.any { it.id == "bias_bias_bias" })
        assertTrue(intermediate.tensors.any { it.id == "output_relu_y" })
        assertEquals(4, intermediate.input.elementCount)
        assertEquals(3, intermediate.output.elementCount)
    }

    @Test
    fun unsupportedPatternFailsWithLoweringDiagnostic() {
        val context = GraphExportContext(backendName = MinervaExportBackend.backendName)

        val exception = assertFailsWith<MinervaLoweringException> {
            MinervaGraphCanonicalizer().convert(activationBeforeLayerGraph(), context)
        }

        assertEquals("minerva.lowering.no_layers", exception.code)
        assertTrue(
            context.diagnostics.any {
                it.stage == GraphExportStage.LOWERING &&
                    it.code == "minerva.lowering.no_layers"
            }
        )
    }
}
