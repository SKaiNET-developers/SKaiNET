package sk.ainet.compile.minerva.examples

import sk.ainet.compile.minerva.MinervaExportFacade
import sk.ainet.compile.minerva.MinervaExportFailureKind
import sk.ainet.compile.minerva.MinervaExportOptions
import sk.ainet.compile.minerva.MinervaHostVerificationMetadata
import sk.ainet.lang.graph.DefaultComputeGraph
import sk.ainet.lang.graph.GraphEdge
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.ops.AddOperation
import sk.ainet.lang.tensor.ops.InputOperation
import sk.ainet.lang.tensor.ops.MatmulOperation
import sk.ainet.lang.tensor.ops.ReluOperation
import sk.ainet.lang.tensor.ops.SigmoidOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType

/**
 * Maintained JVM sample for the Minerva secure MCU export path.
 */
internal object MinervaTinyMlpExportSample {

    @JvmStatic
    public fun main(args: Array<String>): Unit {
        val env = System.getenv()
        val compilerScript = envPath(env, "MINERVA_COMPILER_SCRIPT")
        if (compilerScript == null) {
            println("MINERVA_COMPILER_SCRIPT is not set; running dry validation through NPZ generation.")
        }

        val options = exportOptions(
            compilerScript = compilerScript,
            runtimeRoot = envPath(env, "MINERVA_RUNTIME_ROOT"),
            keyFile = envPath(env, "MINERVA_KEY_FILE"),
            calibrationNpz = envPath(env, "MINERVA_CALIBRATION_NPZ"),
            runCmakeBuild = envFlag(env, "MINERVA_RUN_CMAKE"),
            runCTest = envFlag(env, "MINERVA_RUN_CTEST"),
            hostVerificationTolerance = envFloat(env, "MINERVA_HOST_TOLERANCE"),
            hostOutputPath = envPath(env, "MINERVA_HOST_OUTPUT_PATH"),
            hostAdapterSource = envPath(env, "MINERVA_HOST_ADAPTER_SOURCE"),
            hostIncludeDirs = envPath(env, "MINERVA_HOST_INCLUDE_DIRS"),
            hostLibraryDirs = envPath(env, "MINERVA_HOST_LIBRARY_DIRS"),
            hostLibraries = envPath(env, "MINERVA_HOST_LIBRARIES")
        )
        val result = MinervaExportFacade().exportGraph(tinyMlpGraph(), options)

        println("Minerva export status: ${result.status}")
        result.bundle?.let { bundle ->
            println("Project bundle: ${bundle.outputDir}")
            println("Manifest: ${bundle.manifestPath}")
        }
        result.hostVerification?.let { verification ->
            println("Host verification: ${verification.status}")
        }
        if (compilerScript == null && result.failure?.kind == MinervaExportFailureKind.COMPILER_PREREQUISITE_FAILED) {
            println("Dry validation completed: graph is compatible and model.npz was generated in memory.")
            return
        }
        if (result.failed) {
            error(result.failure?.message ?: "Minerva export failed.")
        }
    }

    internal fun tinyMlpGraph(): DefaultComputeGraph {
        val inputSpec = spec("x", 1, 4)
        val hiddenWeightsSpec = spec("w0", 4, 3, values = linearValues(12, start = 0.10f))
        val hiddenMatmulSpec = spec("matmul0", 1, 3)
        val hiddenBiasSpec = spec("b0", 1, 3, values = linearValues(3, start = 0.01f))
        val hiddenAddSpec = spec("hidden_add", 1, 3)
        val hiddenSpec = spec("hidden", 1, 3)
        val outputWeightsSpec = spec("w1", 3, 2, values = linearValues(6, start = -0.20f))
        val outputMatmulSpec = spec("matmul1", 1, 2)
        val outputBiasSpec = spec("b1", 1, 2, values = linearValues(2, start = -0.03f))
        val outputAddSpec = spec("output_add", 1, 2)
        val outputSpec = spec("y", 1, 2)

        val input = inputNode("input", inputSpec)
        val hiddenWeights = inputNode("hidden_weights", hiddenWeightsSpec)
        val hiddenMatmul = matmulNode("matmul0", inputSpec, hiddenWeightsSpec, hiddenMatmulSpec)
        val hiddenBias = inputNode("hidden_bias", hiddenBiasSpec)
        val hiddenAdd = addNode("hidden_bias_add", hiddenMatmulSpec, hiddenBiasSpec, hiddenAddSpec)
        val hiddenRelu = reluNode("hidden_relu", hiddenAddSpec, hiddenSpec)
        val outputWeights = inputNode("output_weights", outputWeightsSpec)
        val outputMatmul = matmulNode("matmul1", hiddenSpec, outputWeightsSpec, outputMatmulSpec)
        val outputBias = inputNode("output_bias", outputBiasSpec)
        val outputAdd = addNode("output_bias_add", outputMatmulSpec, outputBiasSpec, outputAddSpec)
        val sigmoid = GraphNode(
            id = "output_sigmoid",
            operation = SigmoidOperation<DType, Any>(),
            inputs = listOf(outputAddSpec),
            outputs = listOf(outputSpec)
        )

        return graphOf(
            nodes = listOf(
                input,
                hiddenWeights,
                hiddenMatmul,
                hiddenBias,
                hiddenAdd,
                hiddenRelu,
                outputWeights,
                outputMatmul,
                outputBias,
                outputAdd,
                sigmoid
            ),
            edges = listOf(
                edge("x_to_matmul0", input, hiddenMatmul, inputSpec, destinationInputIndex = 0),
                edge("w0_to_matmul0", hiddenWeights, hiddenMatmul, hiddenWeightsSpec, destinationInputIndex = 1),
                edge("matmul0_to_add", hiddenMatmul, hiddenAdd, hiddenMatmulSpec, destinationInputIndex = 0),
                edge("b0_to_add", hiddenBias, hiddenAdd, hiddenBiasSpec, destinationInputIndex = 1),
                edge("hidden_add_to_relu", hiddenAdd, hiddenRelu, hiddenAddSpec),
                edge("hidden_to_matmul1", hiddenRelu, outputMatmul, hiddenSpec, destinationInputIndex = 0),
                edge("w1_to_matmul1", outputWeights, outputMatmul, outputWeightsSpec, destinationInputIndex = 1),
                edge("matmul1_to_add", outputMatmul, outputAdd, outputMatmulSpec, destinationInputIndex = 0),
                edge("b1_to_add", outputBias, outputAdd, outputBiasSpec, destinationInputIndex = 1),
                edge("output_add_to_sigmoid", outputAdd, sigmoid, outputAddSpec)
            )
        )
    }

    internal fun exportOptions(
        outputDir: String = "build/minerva",
        projectName: String = "TinySecureMlp",
        compilerScript: String? = null,
        runtimeRoot: String? = null,
        keyFile: String? = null,
        calibrationNpz: String? = null,
        runCmakeBuild: Boolean = false,
        runCTest: Boolean = false,
        hostVerificationTolerance: Float? = null,
        hostOutputPath: String? = null,
        hostAdapterSource: String? = null,
        hostIncludeDirs: String? = null,
        hostLibraryDirs: String? = null,
        hostLibraries: String? = null
    ): MinervaExportOptions {
        val metadata = mutableMapOf("sample" to "minerva-tiny-mlp")
        if (runCmakeBuild) {
            metadata[MinervaHostVerificationMetadata.RUN_CMAKE_BUILD] = "true"
        }
        if (runCTest) {
            metadata[MinervaHostVerificationMetadata.RUN_CTEST] = "true"
        }
        hostOutputPath?.let {
            metadata[MinervaHostVerificationMetadata.HOST_OUTPUT_PATH] = it
        }
        hostAdapterSource?.let {
            metadata[MinervaHostVerificationMetadata.HOST_ADAPTER_SOURCE] = it
        }
        hostIncludeDirs?.let {
            metadata[MinervaHostVerificationMetadata.HOST_INCLUDE_DIRS] = it
        }
        hostLibraryDirs?.let {
            metadata[MinervaHostVerificationMetadata.HOST_LIBRARY_DIRS] = it
        }
        hostLibraries?.let {
            metadata[MinervaHostVerificationMetadata.HOST_LIBRARIES] = it
        }
        return MinervaExportOptions(
            outputDir = outputDir,
            projectName = projectName,
            runtimeRoot = runtimeRoot,
            compilerScript = compilerScript,
            keyFile = keyFile,
            calibrationNpz = calibrationNpz,
            hostVerificationTolerance = hostVerificationTolerance ?: 1.0e-3f,
            metadata = metadata
        )
    }

    private fun inputNode(id: String, output: TensorSpec): GraphNode {
        return GraphNode(
            id = id,
            operation = InputOperation<DType, Any>(),
            inputs = emptyList(),
            outputs = listOf(output)
        )
    }

    private fun matmulNode(
        id: String,
        left: TensorSpec,
        right: TensorSpec,
        output: TensorSpec
    ): GraphNode {
        return GraphNode(
            id = id,
            operation = MatmulOperation<DType, Any>(),
            inputs = listOf(left, right),
            outputs = listOf(output)
        )
    }

    private fun addNode(
        id: String,
        left: TensorSpec,
        right: TensorSpec,
        output: TensorSpec
    ): GraphNode {
        return GraphNode(
            id = id,
            operation = AddOperation<DType, Any>(),
            inputs = listOf(left, right),
            outputs = listOf(output)
        )
    }

    private fun reluNode(id: String, input: TensorSpec, output: TensorSpec): GraphNode {
        return GraphNode(
            id = id,
            operation = ReluOperation<DType, Any>(),
            inputs = listOf(input),
            outputs = listOf(output)
        )
    }

    private fun spec(name: String, vararg shape: Int, values: List<Float>? = null): TensorSpec {
        val metadata: Map<String, Any> = values?.let { mapOf("values" to it.toFloatArray()) } ?: emptyMap()
        return TensorSpec(name, shape.toList(), "Float32", metadata = metadata)
    }

    private fun linearValues(count: Int, start: Float): List<Float> {
        return List(count) { index -> start + (index * 0.05f) }
    }

    private fun graphOf(nodes: List<GraphNode>, edges: List<GraphEdge>): DefaultComputeGraph {
        val graph = DefaultComputeGraph()
        nodes.forEach { graph.addNode(it) }
        edges.forEach { graph.addEdge(it) }
        return graph
    }

    private fun edge(
        id: String,
        source: GraphNode,
        destination: GraphNode,
        spec: TensorSpec,
        destinationInputIndex: Int = 0
    ): GraphEdge {
        return GraphEdge(
            id = id,
            source = source,
            destination = destination,
            destinationInputIndex = destinationInputIndex,
            tensorSpec = spec
        )
    }

    private fun envPath(env: Map<String, String>, name: String): String? {
        return env[name]?.trim()?.takeIf { it.isNotEmpty() }
    }

    private fun envFlag(env: Map<String, String>, name: String): Boolean {
        return env[name]?.equals("true", ignoreCase = true) == true
    }

    private fun envFloat(env: Map<String, String>, name: String): Float? {
        return env[name]?.trim()?.takeIf { it.isNotEmpty() }?.toFloat()
    }
}
