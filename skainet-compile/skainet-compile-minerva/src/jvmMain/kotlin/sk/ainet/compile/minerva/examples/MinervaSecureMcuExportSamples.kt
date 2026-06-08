package sk.ainet.compile.minerva.examples

import sk.ainet.compile.export.GraphExportStatus
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
import sk.ainet.lang.tensor.ops.TanhOperation
import sk.ainet.lang.tensor.ops.TensorSpec
import sk.ainet.lang.types.DType

/**
 * Runnable SKaiNET examples inspired by libminerva's secure MCU quickstart
 * and ATmega328P sensor classification demo.
 */
internal object MinervaSecureMcuExportSamples {

    @JvmStatic
    fun main(args: Array<String>) {
        val selected = args.map { it.trim().lowercase() }.filter { it.isNotEmpty() }.toSet()
        val scenarios = scenarios().filter { scenario ->
            selected.isEmpty() || scenario.id.lowercase() in selected
        }
        require(scenarios.isNotEmpty()) {
            "Unknown Minerva example '${args.joinToString(" ")}'. Available: ${
                scenarios().joinToString { it.id }
            }"
        }

        val env = System.getenv()
        scenarios.forEach { scenario ->
            runScenario(scenario, env)
        }
    }

    internal fun scenarios(): List<MinervaExampleScenario> {
        return listOf(sensorClassifier(), safetyGuard())
    }

    internal fun sensorClassifier(): MinervaExampleScenario {
        return MinervaExampleScenario(
            id = "sensor-classifier",
            projectName = "SecureSensorClassifier",
            description = "8-feature MCU sensor classifier with four output classes.",
            graph = sequentialMlpGraph(
                inputName = "sensor_q8_window",
                inputWidth = 8,
                layers = listOf(
                    DenseLayerSpec("hidden0", outputWidth = 16, activation = ExampleActivation.RELU, weightStart = -0.18f),
                    DenseLayerSpec("hidden1", outputWidth = 8, activation = ExampleActivation.RELU, weightStart = 0.12f),
                    DenseLayerSpec("class_logits", outputWidth = 4, activation = ExampleActivation.SIGMOID, weightStart = -0.07f)
                )
            ),
            labels = listOf("idle", "warmup", "nominal", "service"),
            notes = listOf(
                "A0-A3 can feed the first four input slots after ADC-to-Q8 scaling.",
                "Remaining slots can hold rolling deltas, averages, or zero padding.",
                "Firmware can map argmax output classes to LEDs, relays, or telemetry states."
            )
        )
    }

    internal fun safetyGuard(): MinervaExampleScenario {
        return MinervaExampleScenario(
            id = "safety-guard",
            projectName = "SecureSafetyGuard",
            description = "Small health classifier for protect / warn / allow decisions.",
            graph = sequentialMlpGraph(
                inputName = "health_window",
                inputWidth = 6,
                layers = listOf(
                    DenseLayerSpec("feature_mix", outputWidth = 10, activation = ExampleActivation.TANH, weightStart = 0.09f),
                    DenseLayerSpec("guard_hidden", outputWidth = 4, activation = ExampleActivation.RELU, weightStart = -0.14f),
                    DenseLayerSpec("guard_logits", outputWidth = 3, activation = ExampleActivation.SIGMOID, weightStart = 0.04f)
                )
            ),
            labels = listOf("protect", "warn", "allow"),
            notes = listOf(
                "Inputs can represent temperature, voltage, current, vibration, and two rolling features.",
                "Use host verification whenever calibration, keys, compiler, or runtime changes.",
                "Keep real keys outside the generated bundle and source control."
            )
        )
    }

    internal fun exportOptions(
        scenario: MinervaExampleScenario,
        env: Map<String, String> = emptyMap()
    ): MinervaExportOptions {
        val runCmakeBuild = envFlag(env, "MINERVA_RUN_CMAKE")
        val runCTest = envFlag(env, "MINERVA_RUN_CTEST")
        val metadata = mutableMapOf(
            "sample" to "minerva-${scenario.id}",
            "sampleDescription" to scenario.description,
            "classLabels" to scenario.labels.joinToString("|"),
            "sourceShape" to "skainet-compute-graph",
            "runtimePattern" to "libminerva-secure-mcu"
        )
        if (runCmakeBuild) metadata[MinervaHostVerificationMetadata.RUN_CMAKE_BUILD] = "true"
        if (runCTest) metadata[MinervaHostVerificationMetadata.RUN_CTEST] = "true"
        envPath(env, "MINERVA_HOST_OUTPUT_PATH")?.let {
            metadata[MinervaHostVerificationMetadata.HOST_OUTPUT_PATH] = it
        }
        envPath(env, "MINERVA_HOST_ADAPTER_SOURCE")?.let {
            metadata[MinervaHostVerificationMetadata.HOST_ADAPTER_SOURCE] = it
        }
        envPath(env, "MINERVA_HOST_INCLUDE_DIRS")?.let {
            metadata[MinervaHostVerificationMetadata.HOST_INCLUDE_DIRS] = it
        }
        envPath(env, "MINERVA_HOST_LIBRARY_DIRS")?.let {
            metadata[MinervaHostVerificationMetadata.HOST_LIBRARY_DIRS] = it
        }
        envPath(env, "MINERVA_HOST_LIBRARIES")?.let {
            metadata[MinervaHostVerificationMetadata.HOST_LIBRARIES] = it
        }

        return MinervaExportOptions(
            outputDir = "build/minerva-examples",
            projectName = scenario.projectName,
            runtimeRoot = envPath(env, "MINERVA_RUNTIME_ROOT"),
            compilerScript = envPath(env, "MINERVA_COMPILER_SCRIPT"),
            keyFile = envPath(env, "MINERVA_KEY_FILE"),
            calibrationNpz = envPath(env, "MINERVA_CALIBRATION_NPZ"),
            hostVerificationTolerance = envFloat(env, "MINERVA_HOST_TOLERANCE") ?: 1.0e-3f,
            metadata = metadata
        )
    }

    private fun runScenario(scenario: MinervaExampleScenario, env: Map<String, String>) {
        val options = exportOptions(scenario, env)
        val result = MinervaExportFacade().exportGraph(scenario.graph, options)

        println("Minerva example: ${scenario.id}")
        println("Description: ${scenario.description}")
        println("Labels: ${scenario.labels.joinToString(", ")}")
        scenario.notes.forEach { note -> println("Note: $note") }
        println("Export status: ${result.status}")
        result.compatibilityReport?.let { report ->
            println("Layers: ${report.layerCount}")
            println("Estimated SRAM bytes: ${report.estimatedSramBytes}")
        }
        result.bundle?.let { bundle ->
            println("Project bundle: ${bundle.outputDir}")
            println("Manifest: ${bundle.manifestPath}")
        }
        result.hostVerification?.let { verification ->
            println("Host verification: ${verification.status}")
        }
        if (result.status == GraphExportStatus.FAILED &&
            result.failure?.kind == MinervaExportFailureKind.COMPILER_PREREQUISITE_FAILED
        ) {
            println("Dry validation completed: graph is compatible and model.npz was generated in memory.")
            println()
            return
        }
        if (result.failed) {
            error(result.failure?.message ?: "Minerva example '${scenario.id}' failed.")
        }
        println()
    }

    private fun sequentialMlpGraph(
        inputName: String,
        inputWidth: Int,
        layers: List<DenseLayerSpec>
    ): DefaultComputeGraph {
        require(inputWidth > 0) { "inputWidth must be positive" }
        require(layers.isNotEmpty()) { "at least one layer is required" }

        val nodes = mutableListOf<GraphNode>()
        val edges = mutableListOf<GraphEdge>()
        val inputSpec = spec(inputName, 1, inputWidth)
        val input = inputNode("input", inputSpec)
        nodes += input

        var producer = input
        var producerSpec = inputSpec
        var inputSlotWidth = inputWidth

        layers.forEachIndexed { index, layer ->
            val layerPrefix = "${index}_${layer.id}"
            val weightSpec = spec(
                name = "${layer.id}_weights",
                inputSlotWidth,
                layer.outputWidth,
                values = patternedValues(inputSlotWidth * layer.outputWidth, layer.weightStart)
            )
            val weight = inputNode("${layerPrefix}_weights", weightSpec)
            val matmulSpec = spec("${layer.id}_matmul", 1, layer.outputWidth)
            val matmul = matmulNode("${layerPrefix}_matmul", producerSpec, weightSpec, matmulSpec)
            val biasSpec = spec(
                name = "${layer.id}_bias",
                1,
                layer.outputWidth,
                values = patternedValues(layer.outputWidth, start = layer.weightStart / 3.0f)
            )
            val bias = inputNode("${layerPrefix}_bias", biasSpec)
            val biasedSpec = spec("${layer.id}_biased", 1, layer.outputWidth)
            val add = addNode("${layerPrefix}_bias_add", matmulSpec, biasSpec, biasedSpec)

            nodes += listOf(weight, matmul, bias, add)
            edges += edge("${producer.id}_to_${matmul.id}", producer, matmul, producerSpec, destinationInputIndex = 0)
            edges += edge("${weight.id}_to_${matmul.id}", weight, matmul, weightSpec, destinationInputIndex = 1)
            edges += edge("${matmul.id}_to_${add.id}", matmul, add, matmulSpec, destinationInputIndex = 0)
            edges += edge("${bias.id}_to_${add.id}", bias, add, biasSpec, destinationInputIndex = 1)

            val activated = activationNode(
                id = "${layerPrefix}_${layer.activation.id}",
                activation = layer.activation,
                input = biasedSpec,
                output = spec(layer.outputName(index == layers.lastIndex), 1, layer.outputWidth)
            )
            if (activated == null) {
                producer = add
                producerSpec = biasedSpec
            } else {
                nodes += activated
                edges += edge("${add.id}_to_${activated.id}", add, activated, biasedSpec)
                producer = activated
                producerSpec = activated.outputs.single()
            }
            inputSlotWidth = layer.outputWidth
        }

        return graphOf(nodes, edges)
    }

    private fun activationNode(
        id: String,
        activation: ExampleActivation,
        input: TensorSpec,
        output: TensorSpec
    ): GraphNode? {
        val operation = when (activation) {
            ExampleActivation.LINEAR -> return null
            ExampleActivation.RELU -> ReluOperation<DType, Any>()
            ExampleActivation.SIGMOID -> SigmoidOperation<DType, Any>()
            ExampleActivation.TANH -> TanhOperation<DType, Any>()
        }
        return GraphNode(
            id = id,
            operation = operation,
            inputs = listOf(input),
            outputs = listOf(output)
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

    private fun spec(name: String, vararg shape: Int, values: List<Float>? = null): TensorSpec {
        val metadata: Map<String, Any> = values?.let { mapOf("values" to it.toFloatArray()) } ?: emptyMap()
        return TensorSpec(name, shape.toList(), "Float32", metadata = metadata)
    }

    private fun patternedValues(count: Int, start: Float): List<Float> {
        return List(count) { index ->
            val wave = ((index % 7) - 3) * 0.025f
            start + wave + (index / 7) * 0.003f
        }
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

internal data class MinervaExampleScenario(
    val id: String,
    val projectName: String,
    val description: String,
    val graph: DefaultComputeGraph,
    val labels: List<String>,
    val notes: List<String>
)

private data class DenseLayerSpec(
    val id: String,
    val outputWidth: Int,
    val activation: ExampleActivation,
    val weightStart: Float
) {
    init {
        require(id.isNotBlank()) { "id cannot be blank" }
        require(outputWidth > 0) { "outputWidth must be positive" }
    }

    fun outputName(isLast: Boolean): String {
        return if (isLast) "y" else "${id}_${activation.id}"
    }
}

private enum class ExampleActivation(val id: String) {
    LINEAR("linear"),
    RELU("relu"),
    SIGMOID("sigmoid"),
    TANH("tanh")
}
