package sk.ainet.compile.minerva

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import sk.ainet.lang.graph.DefaultComputeGraph

class MinervaCompatibilityValidatorTest {

    @Test
    fun supportedMlpGraphPassesCompatibilityValidation() {
        val report = MinervaCompatibilityValidator().validate(
            graph = validMinervaMlpGraph(),
            options = minervaTestOptions()
        )

        assertTrue(report.compatible)
        assertFalse(report.failed)
        assertEquals(1, report.layerCount)
        assertTrue(report.issues.isEmpty())
        assertTrue(report.estimatedSramBytes > 0)
        assertTrue(report.diagnostics.infos.any { it.code == "minerva.compatibility.passed" })
    }

    @Test
    fun emptyGraphFailsBeforeCompilerInvocation() {
        val report = MinervaCompatibilityValidator().validate(
            graph = DefaultComputeGraph(),
            options = minervaTestOptions()
        )

        assertFalse(report.compatible)
        assertTrue(report.diagnostics.hasErrors)
        assertTrue(
            report.issues.any {
                it.kind == MinervaCompatibilityIssueKind.GRAPH_VALIDATION &&
                    it.code == "minerva.compatibility.empty_graph"
            }
        )
    }

    @Test
    fun unsupportedOperationNamesNodeAndRemediation() {
        val report = MinervaCompatibilityValidator().validate(
            graph = unsupportedMinervaOperationGraph(),
            options = minervaTestOptions()
        )
        val issue = report.issues.first {
            it.kind == MinervaCompatibilityIssueKind.UNSUPPORTED_OPERATION
        }

        assertFalse(report.compatible)
        assertEquals("conv", issue.nodeId)
        assertEquals("conv1d", issue.operationName)
        assertTrue(issue.remediation.contains("sequential MLP"))
    }

    @Test
    fun unsupportedTopologyNamesBranchingNode() {
        val report = MinervaCompatibilityValidator().validate(
            graph = branchingMinervaGraph(),
            options = minervaTestOptions()
        )
        val issue = report.issues.first {
            it.kind == MinervaCompatibilityIssueKind.UNSUPPORTED_TOPOLOGY &&
                it.code == "minerva.compatibility.branching"
        }

        assertFalse(report.compatible)
        assertEquals("input", issue.nodeId)
        assertEquals("2", issue.details["consumerCount"])
    }

    @Test
    fun missingStaticShapesFailCompatibilityValidation() {
        val report = MinervaCompatibilityValidator().validate(
            graph = missingShapeMinervaGraph(),
            options = minervaTestOptions()
        )

        assertFalse(report.compatible)
        assertTrue(
            report.issues.any {
                it.kind == MinervaCompatibilityIssueKind.MISSING_SHAPE &&
                    it.nodeId == "input"
            }
        )
        assertTrue(report.diagnostics.errors.any { it.code == "minerva.compatibility.missing_shape" })
    }

    @Test
    fun activationPlacementMustFollowSupportedLayerPattern() {
        val report = MinervaCompatibilityValidator().validate(
            graph = activationBeforeLayerGraph(),
            options = minervaTestOptions()
        )
        val issue = report.issues.first {
            it.kind == MinervaCompatibilityIssueKind.INCOMPATIBLE_ACTIVATION_PLACEMENT
        }

        assertFalse(report.compatible)
        assertEquals("relu", issue.nodeId)
        assertEquals("input", issue.details["producer"])
    }

    @Test
    fun targetMemoryOverflowFailsCompatibilityValidation() {
        val report = MinervaCompatibilityValidator().validate(
            graph = validMinervaMlpGraph(inputWidth = 2048, outputWidth = 16),
            options = minervaTestOptions()
        )
        val issue = report.issues.first {
            it.kind == MinervaCompatibilityIssueKind.MEMORY_BUDGET_EXCEEDED
        }

        assertFalse(report.compatible)
        assertEquals("2048", issue.details["targetSramBytes"])
        assertTrue(report.estimatedSramBytes > MinervaTarget.ATMEGA328P.sramBytes)
    }
}
