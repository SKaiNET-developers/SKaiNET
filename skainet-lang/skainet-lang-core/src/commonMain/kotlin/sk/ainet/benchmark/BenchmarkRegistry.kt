package sk.ainet.benchmark

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.benchmark.SlicingBenchmarkConfig
import sk.ainet.lang.tensor.benchmark.TensorFactory
import sk.ainet.lang.tensor.benchmark.slicingBenchmarkSuite
import sk.ainet.lang.types.DType

public object BenchmarkSuites {

    public fun <T : DType, V> slicing(
        config: SlicingBenchmarkConfig = SlicingBenchmarkConfig(),
        contextFactory: () -> ExecutionContext,
        tensorFactory: TensorFactory<T, V>
    ): BenchmarkSuite = slicingBenchmarkSuite(
        config = config,
        contextFactory = contextFactory,
        tensorFactory = tensorFactory
    )
}

public fun <T : DType, V> runSlicingBenchmarks(
    config: SlicingBenchmarkConfig = SlicingBenchmarkConfig(),
    contextFactory: () -> ExecutionContext,
    tensorFactory: TensorFactory<T, V>
): BenchmarkSuiteResult {
    val suite = BenchmarkSuites.slicing(config, contextFactory, tensorFactory)
    return BenchmarkRunner.runSuite(suite)
}
