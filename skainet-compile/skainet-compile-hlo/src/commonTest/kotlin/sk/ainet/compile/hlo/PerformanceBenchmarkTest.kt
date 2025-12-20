package sk.ainet.compile.hlo

import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.time.Duration
import sk.ainet.compile.hlo.validation.*

class PerformanceBenchmarkTest {
    
    private val benchmark = PerformanceBenchmark()
    private val testUtilities = MlirTestUtilities()
    
    @Test
    fun benchmarkConversion_simpleGraph_returnsMetrics() {
        val graph = testUtilities.createSimpleTestGraph()
        val converter = StableHloConverterFactory.createBasic()
        
        val metrics = benchmark.benchmarkConversion(converter, graph, "test")
        
        assertTrue(metrics.conversionTime >= Duration.ZERO, "Conversion time should be non-negative")
        assertTrue(metrics.validationTime >= Duration.ZERO, "Validation time should be non-negative")
        assertTrue(metrics.totalTime >= metrics.conversionTime, "Total time should be >= conversion time")
        assertTrue(metrics.moduleSize > 0, "Module size should be positive")
        assertTrue(metrics.operationCount >= 0, "Operation count should be non-negative")
        assertTrue(metrics.ssaValueCount >= 0, "SSA value count should be non-negative")
    }
    
    @Test
    fun benchmarkMultipleRuns_returnsAggregatedResults() {
        val graph = testUtilities.createSimpleTestGraph()
        val converter = StableHloConverterFactory.createBasic()
        val runs = 3
        
        val results = benchmark.benchmarkMultipleRuns(converter, graph, runs, "MultiRun Test")
        
        assertEquals("MultiRun Test", results.testName)
        assertEquals(runs, results.runs.size, "Should have correct number of runs")
        assertNotNull(results.averageMetrics, "Should have average metrics")
        assertNotNull(results.minMetrics, "Should have min metrics")
        assertNotNull(results.maxMetrics, "Should have max metrics")
        
        // Verify that min <= average <= max for total time (allowing for small timing variations)
        assertTrue(results.minMetrics.totalTime.inWholeMilliseconds <= results.averageMetrics.totalTime.inWholeMilliseconds + 1, 
                  "Min time should be <= average time (${results.minMetrics.totalTime} vs ${results.averageMetrics.totalTime})")
        assertTrue(results.averageMetrics.totalTime.inWholeMilliseconds <= results.maxMetrics.totalTime.inWholeMilliseconds + 1, 
                  "Average time should be <= max time (${results.averageMetrics.totalTime} vs ${results.maxMetrics.totalTime})")
    }
    
    @Test
    fun benchmarkThroughput_returnsValidMetrics() {
        val graph = testUtilities.createSimpleTestGraph()
        val graphs = listOf(graph, graph, graph) // Multiple copies for throughput testing
        val converter = StableHloConverterFactory.createBasic()
        val timeLimit = Duration.parse("1s")
        
        val throughputMetrics = benchmark.benchmarkThroughput(converter, graphs, timeLimit)
        
        assertTrue(throughputMetrics.conversionsCompleted >= 0, "Conversions completed should be non-negative")
        assertTrue(throughputMetrics.totalOperations >= 0, "Total operations should be non-negative")
        assertTrue(throughputMetrics.actualTime <= timeLimit * 1.1, "Actual time should be close to time limit")
        assertTrue(throughputMetrics.conversionsPerSecond >= 0.0, "Conversions per second should be non-negative")
        assertTrue(throughputMetrics.operationsPerSecond >= 0.0, "Operations per second should be non-negative")
    }
    
    @Test
    fun benchmarkMemoryUsage_returnsMetrics() {
        val graph = testUtilities.createSimpleTestGraph()
        val converter = StableHloConverterFactory.createBasic()
        
        val memoryMetrics = benchmark.benchmarkMemoryUsage(converter, graph)
        
        assertTrue(memoryMetrics.moduleSize > 0, "Module size should be positive")
        // Note: Memory usage estimation is platform-dependent and may not be accurate
        assertNotNull(memoryMetrics.beforeConversion, "Should have before conversion measurement")
        assertNotNull(memoryMetrics.afterConversion, "Should have after conversion measurement")
    }
    
    @Test
    fun compareConverters_returnsComparisonResults() {
        val graph = testUtilities.createSimpleTestGraph()
        val converter1 = StableHloConverterFactory.createBasic()
        val converter2 = StableHloConverterFactory.createBasic() // Same converter for testing
        
        val converters = mapOf(
            "Converter1" to converter1,
            "Converter2" to converter2
        )
        
        val results = benchmark.compareConverters(converters, graph, runs = 2)
        
        assertEquals(2, results.size, "Should have results for both converters")
        assertTrue(results.containsKey("Converter1"), "Should have results for Converter1")
        assertTrue(results.containsKey("Converter2"), "Should have results for Converter2")
        
        for ((name, result) in results) {
            assertEquals(2, result.runs.size, "Each converter should have 2 runs")
            assertEquals(name, result.testName, "Test name should match converter name")
        }
    }
    
    @Test
    fun generatePerformanceReport_returnsFormattedReport() {
        val graph = testUtilities.createSimpleTestGraph()
        val converter = StableHloConverterFactory.createBasic()
        
        val results = mapOf(
            "TestConverter" to benchmark.benchmarkMultipleRuns(converter, graph, 2, "TestConverter")
        )
        
        val report = benchmark.generatePerformanceReport(results)
        
        assertTrue(report.isNotEmpty(), "Report should not be empty")
        assertTrue(report.contains("Performance Report"), "Report should have title")
        assertTrue(report.contains("TestConverter"), "Report should mention converter name")
        assertTrue(report.contains("Runs: 2"), "Report should mention number of runs")
    }
    
    @Test
    fun conversionMetrics_toMap_returnsValidMap() {
        val metrics = ConversionMetrics(
            conversionTime = Duration.parse("10ms"),
            validationTime = Duration.parse("5ms"),
            totalTime = Duration.parse("15ms"),
            moduleSize = 1000,
            operationCount = 5,
            ssaValueCount = 10
        )
        
        val map = metrics.toMap()
        
        assertEquals(10L, map["conversion_time_ms"])
        assertEquals(5L, map["validation_time_ms"])
        assertEquals(15L, map["total_time_ms"])
        assertEquals(1000, map["module_size_bytes"])
        assertEquals(5, map["operation_count"])
        assertEquals(10, map["ssa_value_count"])
    }
    
    @Test
    fun throughputMetrics_summary_returnsFormattedSummary() {
        val metrics = ThroughputMetrics(
            conversionsCompleted = 100,
            totalOperations = 500,
            actualTime = Duration.parse("10s"),
            conversionsPerSecond = 10.0,
            operationsPerSecond = 50.0
        )
        
        val summary = metrics.summary()
        
        assertTrue(summary.contains("Throughput Metrics"), "Summary should have title")
        assertTrue(summary.contains("100"), "Summary should mention conversions completed")
        assertTrue(summary.contains("500"), "Summary should mention total operations")
        assertTrue(summary.contains("Conversions per second"), "Summary should mention conversions per second")
        assertTrue(summary.contains("Operations per second"), "Summary should mention operations per second")
    }
    
    @Test
    fun memoryMetrics_summary_returnsFormattedSummary() {
        val metrics = MemoryMetrics(
            beforeConversion = 1000L,
            afterConversion = 1500L,
            estimatedUsage = 500L,
            moduleSize = 200L
        )
        
        val summary = metrics.summary()
        
        assertTrue(summary.contains("Memory Metrics"), "Summary should have title")
        assertTrue(summary.contains("1000"), "Summary should mention before conversion")
        assertTrue(summary.contains("1500"), "Summary should mention after conversion")
        assertTrue(summary.contains("500"), "Summary should mention estimated usage")
        assertTrue(summary.contains("200"), "Summary should mention module size")
    }
    
    @Test
    fun benchmarkResults_summary_returnsFormattedSummary() {
        val graph = testUtilities.createSimpleTestGraph()
        val converter = StableHloConverterFactory.createBasic()
        
        val results = benchmark.benchmarkMultipleRuns(converter, graph, 2, "Test Summary")
        val summary = results.summary()
        
        assertTrue(summary.contains("Test Summary"), "Summary should contain test name")
        assertTrue(summary.contains("Runs: 2"), "Summary should mention number of runs")
        assertTrue(summary.contains("Average conversion time"), "Summary should mention average conversion time")
        assertTrue(summary.contains("Min total time"), "Summary should mention min time")
        assertTrue(summary.contains("Max total time"), "Summary should mention max time")
    }
}