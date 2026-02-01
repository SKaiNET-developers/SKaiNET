package sk.ainet.lang.tensor.benchmark

import sk.ainet.benchmark.BenchmarkCaseBuilder
import sk.ainet.benchmark.BenchmarkMetric
import sk.ainet.benchmark.BenchmarkSuite
import sk.ainet.benchmark.benchmarkSuite
import sk.ainet.benchmark.BenchmarkExecutionScope
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.sliceView
import sk.ainet.lang.tensor.materialize
import sk.ainet.lang.types.DType
import kotlin.random.Random

public data class SlicingBenchmarkConfig(
    val warmupIterations: Int = 5,
    val iterations: Int = 20,
    val tensorShapes: List<IntArray> = listOf(
        intArrayOf(32, 128, 224, 224),
        intArrayOf(1024, 768),
        intArrayOf(8, 8, 512, 512),
        intArrayOf(100, 1000)
    ),
    val accessSamples: Int = 100
)

public typealias TensorFactory<T, V> = (ExecutionContext, IntArray) -> Tensor<T, V>

public fun <T : DType, V> slicingBenchmarkSuite(
    name: String = "tensor-slicing",
    config: SlicingBenchmarkConfig = SlicingBenchmarkConfig(),
    contextFactory: () -> ExecutionContext,
    tensorFactory: TensorFactory<T, V>
): BenchmarkSuite =
    benchmarkSuite(name) {
        description("Slicing benchmarks covering view creation, access patterns, and ML scenarios.")
        context(contextFactory)
        metrics {
            latency()
            memory()
        }

        config.tensorShapes.forEach { shape ->
            val shapeLabel = shape.joinToString("x")
            case("view-vs-copy-$shapeLabel") {
                warmup(config.warmupIterations)
                iterations(config.iterations)
                description("View creation patterns for tensor shape $shapeLabel")
                setupTensor(shape, tensorFactory)
                runViewVsCopy<T,V>()
            }

            case("access-patterns-$shapeLabel") {
                warmup(config.warmupIterations)
                iterations(config.iterations)
                description("Random element access through slicing views for tensor shape $shapeLabel")
                setupTensor(shape, tensorFactory)
                runAccessPatterns<T,V>(config)
            }

            case("ml-scenarios-$shapeLabel") {
                warmup(config.warmupIterations)
                iterations(config.iterations)
                description("Representative ML slicing scenarios for tensor shape $shapeLabel")
                setupTensor(shape, tensorFactory)
                runMlScenarios<T,V>(config)
            }
        }
    }

private fun <T : DType, V> BenchmarkCaseBuilder.setupTensor(
    shape: IntArray,
    factory: TensorFactory<T, V>
) {
    setup {
        val tensor = factory(context, shape)
        remember("tensor", tensor)
    }
}

@Suppress("UNCHECKED_CAST")
private fun <T : DType, V> BenchmarkExecutionScope.tensor(): Tensor<T, V> =
    recall("tensor")

private fun <T : DType, V> BenchmarkCaseBuilder.runViewVsCopy() {
    run {
        val tensor = tensor<T, V>()
        val slicePatterns = listOf<(Tensor<T, V>) -> sk.ainet.lang.tensor.TensorView<T, V>>(
            { t ->
                t.sliceView {
                    segment { range(0, minOf(8, t.shape[0])) }
                    repeat(t.rank - 1) { segment { all() } }
                }
            },
            { t ->
                if (t.rank >= 2) {
                    t.sliceView {
                        segment { all() }
                        segment { range(0, minOf(64, t.shape[1])) }
                        repeat(t.rank - 2) { segment { all() } }
                    }
                } else {
                    t.sliceView { segment { all() } }
                }
            },
            { t ->
                if (t.rank >= 4) {
                    t.sliceView {
                        segment { all() }
                        segment { all() }
                        segment { range(0, minOf(112, t.shape[2])) }
                        segment { range(0, minOf(112, t.shape[3])) }
                    }
                } else {
                    t.sliceView { repeat(t.rank) { segment { all() } } }
                }
            }
        )

        slicePatterns.forEach { pattern ->
            val view = pattern(tensor)
            view.materialize()
        }
    }
}

private fun <T : DType, V> BenchmarkCaseBuilder.runAccessPatterns(
    config: SlicingBenchmarkConfig
) {
    run {
        val tensor = tensor<T, V>()
        val views = listOf(
            tensor.sliceView {
                segment { range(0, minOf(16, tensor.shape[0])) }
                repeat(tensor.rank - 1) { segment { all() } }
            },
            tensor.sliceView {
                segment { step(0, tensor.shape[0], 2) }
                repeat(tensor.rank - 1) { segment { all() } }
            },
            if (tensor.rank >= 4) {
                tensor.sliceView {
                    segment { all() }
                    segment { all() }
                    segment { range(10, minOf(110, tensor.shape[2])) }
                    segment { range(10, minOf(110, tensor.shape[3])) }
                }
            } else {
                tensor.sliceView { repeat(tensor.rank) { segment { all() } } }
            }
        )
        views.forEach { view ->
            accessRandomElements(view, config.accessSamples)
        }
    }
}

private fun <T : DType, V> BenchmarkCaseBuilder.runMlScenarios(
    config: SlicingBenchmarkConfig
) {
    run {
        val tensor = tensor<T, V>()

        // Mini-batch extraction
        val batchSize = minOf(32, tensor.shape[0])
        val miniBatchView = tensor.sliceView {
            segment { range(0, batchSize) }
            repeat(tensor.rank - 1) { segment { all() } }
        }
        accessRandomElements(miniBatchView, config.accessSamples / 2)

        // Feature-map channels
        if (tensor.rank >= 4) {
            val featureSlice = tensor.sliceView {
                segment { all() }
                segment { range(0, minOf(128, tensor.shape[1])) }
                segment { all() }
                segment { all() }
            }
            accessRandomElements(featureSlice, config.accessSamples / 2)
        }

        // Sliding window along last dimension
        if (tensor.rank >= 2) {
            val windowSize = minOf(64, tensor.shape[tensor.rank - 1])
            val slidingWindow = tensor.sliceView {
                repeat(tensor.rank - 1) { segment { all() } }
                segment { range(0, windowSize) }
            }
            accessRandomElements(slidingWindow, config.accessSamples / 2)
        }
    }
}

private fun <T : DType, V> accessRandomElements(tensor: Tensor<T, V>, count: Int) {
    val shape = tensor.shape
    val random = Random(0)
    repeat(count) {
        val indices = IntArray(shape.rank) { dim ->
            val size = shape[dim]
            if (size == 0) 0 else random.nextInt(size)
        }
        tensor.data.get(*indices)
    }
}
