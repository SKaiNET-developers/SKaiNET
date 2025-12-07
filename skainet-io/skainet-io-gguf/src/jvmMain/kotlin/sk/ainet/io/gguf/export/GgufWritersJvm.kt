package sk.ainet.io.gguf.export

import sk.ainet.lang.graph.ComputeGraph
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.graph.exec.GraphExecutionContext
import java.io.File

/** Write GGUF bytes for a graph + weights directly to a file on JVM. */
public fun writeGraphToGgufFile(
    file: File,
    graph: ComputeGraph,
    weights: Map<String, Tensor<*, *>>,
    label: String = file.nameWithoutExtension,
    options: GgufExportOptions = GgufExportOptions()
): GGUFWriteReport {
    val (report, bytes) = writeGraphToGgufBytes(graph, weights, label, options)
    file.outputStream().use { it.write(bytes) }
    return report
}

/** Write GGUF bytes for a model + forward pass directly to a file on JVM. */
public fun writeModelToGgufFile(
    file: File,
    model: Module<*, *>,
    forwardPass: (GraphExecutionContext) -> Unit,
    label: String = file.nameWithoutExtension,
    options: GgufExportOptions = GgufExportOptions()
): GGUFWriteReport {
    val (report, bytes) = writeModelToGgufBytes(model, forwardPass, label, options)
    file.outputStream().use { it.write(bytes) }
    return report
}
