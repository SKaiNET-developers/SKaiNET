package sk.ainet.compile.hlo.validation

import sk.ainet.compile.hlo.StableHloModule
import sk.ainet.compile.hlo.StableHloConverter
import sk.ainet.lang.graph.ComputeGraph
import kotlin.time.Duration
import kotlin.time.TimeSource

/**
 * Performance metrics for conversion operations
 */
public data class ConversionMetrics(
    val conversionTime: Duration,
    val validationTime: Duration,
    val totalTime: Duration,
    val moduleSize: Int,
    val operationCount: Int,
    val ssaValueCount: Int,
    val memoryUsage: Long? = null
) {
    public fun toMap(): Map<String, Any> {
        return mapOf(
            "conversion_time_ms" to conversionTime.inWholeMilliseconds,
            "validation_time_ms" to validationTime.inWholeMilliseconds,
            "total_time_ms" to totalTime.inWholeMilliseconds,
            "module_size_bytes" to moduleSize,
            "operation_count" to operationCount,
            "ssa_value_count" to ssaValueCount,
            "memory_usage_bytes" to (memoryUsage ?: -1)
        )
    }
}

/**
 * Benchmark results for multiple runs
 */
public data class BenchmarkResults(
    val testName: String,
    val runs: List<ConversionMetrics>,
    val averageMetrics: ConversionMetrics,
    val minMetrics: ConversionMetrics,
    val maxMetrics: ConversionMetrics
) {
    public fun summary(): String {
        return buildString {
            appendLine("Benchmark Results for: $testName")
            appendLine("Runs: ${runs.size}")
            appendLine("Average conversion time: ${averageMetrics.conversionTime}")
            appendLine("Average validation time: ${averageMetrics.validationTime}")
            appendLine("Average total time: ${averageMetrics.totalTime}")
            appendLine("Average module size: ${averageMetrics.moduleSize} bytes")
            appendLine("Average operation count: ${averageMetrics.operationCount}")
            appendLine("Min total time: ${minMetrics.totalTime}")
            appendLine("Max total time: ${maxMetrics.totalTime}")
        }
    }
}

/**
 * Performance benchmarking utilities for StableHLO conversion.
 * 
 * This class provides comprehensive benchmarking capabilities for measuring
 * the performance of StableHLO conversion operations, including timing,
 * memory usage, and throughput metrics.
 */
public class PerformanceBenchmark {
    
    private val testUtilities = MlirTestUtilities()
    
    /**
     * Benchmark a single conversion operation
     */
    public fun benchmarkConversion(
        converter: StableHloConverter,
        graph: ComputeGraph,
        functionName: String = "main"
    ): ConversionMetrics {
        val timeSource = TimeSource.Monotonic
        val startTime = timeSource.markNow()
        
        // Measure conversion time
        val conversionStart = timeSource.markNow()
        val module = converter.convert(graph, functionName)
        val conversionTime = conversionStart.elapsedNow()
        
        // Measure validation time
        val validationStart = timeSource.markNow()
        val verificationResult = testUtilities.verifyCorrectness(module)
        val validationTime = validationStart.elapsedNow()
        
        val totalTime = startTime.elapsedNow()
        
        // Calculate module metrics
        val moduleSize = module.content.encodeToByteArray().size
        val operationCount = countOperations(module.content)
        val ssaValueCount = countSSAValues(module.content)
        
        return ConversionMetrics(
            conversionTime = conversionTime,
            validationTime = validationTime,
            totalTime = totalTime,
            moduleSize = moduleSize,
            operationCount = operationCount,
            ssaValueCount = ssaValueCount
        )
    }
    
    /**
     * Run multiple benchmark iterations and collect statistics
     */
    public fun benchmarkMultipleRuns(
        converter: StableHloConverter,
        graph: ComputeGraph,
        runs: Int = 10,
        testName: String = "Conversion Benchmark"
    ): BenchmarkResults {
        require(runs > 0) { "Number of runs must be positive" }
        
        val metrics = mutableListOf<ConversionMetrics>()
        
        // Warm-up run (not counted in results)
        benchmarkConversion(converter, graph)
        
        // Actual benchmark runs
        repeat(runs) {
            val metric = benchmarkConversion(converter, graph)
            metrics.add(metric)
        }
        
        val averageMetrics = calculateAverageMetrics(metrics)
        val minMetrics = calculateMinMetrics(metrics)
        val maxMetrics = calculateMaxMetrics(metrics)
        
        return BenchmarkResults(
            testName = testName,
            runs = metrics,
            averageMetrics = averageMetrics,
            minMetrics = minMetrics,
            maxMetrics = maxMetrics
        )
    }
    
    /**
     * Benchmark conversion throughput (operations per second)
     */
    public fun benchmarkThroughput(
        converter: StableHloConverter,
        graphs: List<ComputeGraph>,
        timeLimit: Duration = Duration.parse("10s")
    ): ThroughputMetrics {
        val timeSource = TimeSource.Monotonic
        val startTime = timeSource.markNow()
        var conversionsCompleted = 0
        var totalOperations = 0
        
        while (startTime.elapsedNow() < timeLimit && graphs.isNotEmpty()) {
            for (graph in graphs) {
                if (startTime.elapsedNow() >= timeLimit) break
                
                val module = converter.convert(graph)
                conversionsCompleted++
                totalOperations += countOperations(module.content)
            }
        }
        
        val actualTime = startTime.elapsedNow()
        val conversionsPerSecond = conversionsCompleted.toDouble() / actualTime.inWholeSeconds
        val operationsPerSecond = totalOperations.toDouble() / actualTime.inWholeSeconds
        
        return ThroughputMetrics(
            conversionsCompleted = conversionsCompleted,
            totalOperations = totalOperations,
            actualTime = actualTime,
            conversionsPerSecond = conversionsPerSecond,
            operationsPerSecond = operationsPerSecond
        )
    }
    
    /**
     * Benchmark memory usage during conversion
     */
    public fun benchmarkMemoryUsage(
        converter: StableHloConverter,
        graph: ComputeGraph
    ): MemoryMetrics {
        // Note: Kotlin/Multiplatform doesn't have direct memory measurement APIs
        // This is a placeholder implementation that could be enhanced with platform-specific code
        
        val beforeConversion = estimateMemoryUsage()
        val module = converter.convert(graph)
        val afterConversion = estimateMemoryUsage()
        
        return MemoryMetrics(
            beforeConversion = beforeConversion,
            afterConversion = afterConversion,
            estimatedUsage = afterConversion - beforeConversion,
            moduleSize = module.content.encodeToByteArray().size.toLong()
        )
    }
    
    /**
     * Compare performance between different converters
     */
    public fun compareConverters(
        converters: Map<String, StableHloConverter>,
        graph: ComputeGraph,
        runs: Int = 5
    ): Map<String, BenchmarkResults> {
        val results = mutableMapOf<String, BenchmarkResults>()
        
        for ((name, converter) in converters) {
            val benchmarkResult = benchmarkMultipleRuns(converter, graph, runs, name)
            results[name] = benchmarkResult
        }
        
        return results
    }
    
    /**
     * Generate a performance report
     */
    public fun generatePerformanceReport(
        results: Map<String, BenchmarkResults>
    ): String {
        return buildString {
            appendLine("=== StableHLO Conversion Performance Report ===")
            appendLine()
            
            for ((name, result) in results) {
                appendLine(result.summary())
                appendLine()
            }
            
            if (results.size > 1) {
                appendLine("=== Comparison ===")
                val sortedByTime = results.entries.sortedBy { it.value.averageMetrics.totalTime }
                sortedByTime.forEachIndexed { index, (name, result) ->
                    val rank = index + 1
                    appendLine("$rank. $name - ${result.averageMetrics.totalTime}")
                }
            }
        }
    }
    
    private fun calculateAverageMetrics(metrics: List<ConversionMetrics>): ConversionMetrics {
        val avgConversionTimeMs = metrics.map { it.conversionTime.inWholeMilliseconds }.average().toLong()
        val avgValidationTimeMs = metrics.map { it.validationTime.inWholeMilliseconds }.average().toLong()
        val avgTotalTimeMs = metrics.map { it.totalTime.inWholeMilliseconds }.average().toLong()
        val avgModuleSize = metrics.map { it.moduleSize }.average().toInt()
        val avgOperationCount = metrics.map { it.operationCount }.average().toInt()
        val avgSSAValueCount = metrics.map { it.ssaValueCount }.average().toInt()
        
        return ConversionMetrics(
            conversionTime = Duration.parse("${avgConversionTimeMs}ms"),
            validationTime = Duration.parse("${avgValidationTimeMs}ms"),
            totalTime = Duration.parse("${avgTotalTimeMs}ms"),
            moduleSize = avgModuleSize,
            operationCount = avgOperationCount,
            ssaValueCount = avgSSAValueCount
        )
    }
    
    private fun calculateMinMetrics(metrics: List<ConversionMetrics>): ConversionMetrics {
        return metrics.minByOrNull { it.totalTime } ?: metrics.first()
    }
    
    private fun calculateMaxMetrics(metrics: List<ConversionMetrics>): ConversionMetrics {
        return metrics.maxByOrNull { it.totalTime } ?: metrics.first()
    }
    
    private fun countOperations(content: String): Int {
        return content.lines().count { line ->
            line.trim().contains("stablehlo.")
        }
    }
    
    private fun countSSAValues(content: String): Int {
        val regex = Regex("""%[a-zA-Z0-9_]+""")
        return regex.findAll(content).count()
    }
    
    private fun estimateMemoryUsage(): Long {
        // Placeholder implementation - would need platform-specific code for accurate measurement
        return kotlin.random.Random.nextLong(1000000) // Dummy value
    }
}

/**
 * Throughput metrics for conversion operations
 */
public data class ThroughputMetrics(
    val conversionsCompleted: Int,
    val totalOperations: Int,
    val actualTime: Duration,
    val conversionsPerSecond: Double,
    val operationsPerSecond: Double
) {
    public fun summary(): String {
        return buildString {
            appendLine("Throughput Metrics:")
            appendLine("  Conversions completed: $conversionsCompleted")
            appendLine("  Total operations: $totalOperations")
            appendLine("  Time elapsed: $actualTime")
            appendLine("  Conversions per second: ${conversionsPerSecond}")
            appendLine("  Operations per second: ${operationsPerSecond}")
        }
    }
}

/**
 * Memory usage metrics
 */
public data class MemoryMetrics(
    val beforeConversion: Long,
    val afterConversion: Long,
    val estimatedUsage: Long,
    val moduleSize: Long
) {
    public fun summary(): String {
        return buildString {
            appendLine("Memory Metrics:")
            appendLine("  Before conversion: $beforeConversion bytes")
            appendLine("  After conversion: $afterConversion bytes")
            appendLine("  Estimated usage: $estimatedUsage bytes")
            appendLine("  Module size: $moduleSize bytes")
        }
    }
}