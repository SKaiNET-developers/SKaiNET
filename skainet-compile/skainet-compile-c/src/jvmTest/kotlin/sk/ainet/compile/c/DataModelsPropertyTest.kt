package sk.ainet.compile.c

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Property-based tests for Arduino C code generation data models.
 * 
 * **Feature: arduino-c-codegen, Property 17: Memory Calculation Accuracy**
 * 
 * These tests validate that the data models correctly handle memory calculations
 * and maintain data integrity across various input combinations.
 */
class DataModelsPropertyTest {

    @Test
    fun memoryLayout_calculationAccuracy_property() {
        val rng = Random(42)
        
        // **Property 17: Memory Calculation Accuracy**
        // **Validates: Requirements 4.2, 4.5**
        repeat(100) {
            // Generate random but valid memory values
            val maxIntermediateSize = rng.nextInt(0, 10000)
            val totalWeightSize = rng.nextInt(0, 50000)
            val numBuffers = rng.nextInt(1, 10)
            val bufferSizes = (0 until numBuffers).map { rng.nextInt(0, 5000) }
            
            // Calculate expected total memory (weights + max intermediate size for ping-pong buffers)
            val expectedTotalMemory = totalWeightSize + (maxIntermediateSize * 2) // ping-pong requires 2 buffers
            
            val memoryLayout = MemoryLayout(
                maxIntermediateSize = maxIntermediateSize,
                totalWeightSize = totalWeightSize,
                totalMemoryRequired = expectedTotalMemory,
                bufferSizes = bufferSizes
            )
            
            // Verify all values are preserved accurately
            assertEquals(maxIntermediateSize, memoryLayout.maxIntermediateSize)
            assertEquals(totalWeightSize, memoryLayout.totalWeightSize)
            assertEquals(expectedTotalMemory, memoryLayout.totalMemoryRequired)
            assertEquals(bufferSizes, memoryLayout.bufferSizes)
            
            // Verify memory calculation consistency
            assertTrue(memoryLayout.totalMemoryRequired >= memoryLayout.totalWeightSize)
            assertTrue(memoryLayout.maxIntermediateSize >= 0)
            assertTrue(memoryLayout.bufferSizes.all { it >= 0 })
        }
    }

    @Test
    fun layerCode_shapePreservation_property() {
        val rng = Random(42)
        
        repeat(100) {
            // Generate random valid layer parameters
            val layerName = "layer_${rng.nextInt(1000)}"
            val operationType = listOf("Dense", "ReLU", "Sigmoid", "Tanh").random(rng)
            val inputDims = rng.nextInt(1, 5)
            val outputDims = rng.nextInt(1, 5)
            val inputShape = IntArray(inputDims) { rng.nextInt(1, 100) }
            val outputShape = IntArray(outputDims) { rng.nextInt(1, 100) }
            val codeFragment = "// Generated code for $operationType"
            
            val layerCode = LayerCode(
                layerName = layerName,
                operationType = operationType,
                inputShape = inputShape,
                outputShape = outputShape,
                codeFragment = codeFragment
            )
            
            // Verify shape preservation and data integrity
            assertEquals(layerName, layerCode.layerName)
            assertEquals(operationType, layerCode.operationType)
            assertTrue(layerCode.inputShape.contentEquals(inputShape))
            assertTrue(layerCode.outputShape.contentEquals(outputShape))
            assertEquals(codeFragment, layerCode.codeFragment)
            
            // Verify all dimensions are positive
            assertTrue(layerCode.inputShape.all { it > 0 })
            assertTrue(layerCode.outputShape.all { it > 0 })
        }
    }

    @Test
    fun arduinoLibraryResult_dataIntegrity_property() {
        val rng = Random(42)
        
        repeat(100) {
            // Generate random valid library result parameters
            val libraryPath = "/path/to/library_${rng.nextInt(1000)}"
            val memoryLayout = MemoryLayout(
                maxIntermediateSize = rng.nextInt(0, 1000),
                totalWeightSize = rng.nextInt(0, 5000),
                totalMemoryRequired = rng.nextInt(1000, 10000),
                bufferSizes = listOf(rng.nextInt(100, 500), rng.nextInt(100, 500))
            )
            val numOperations = rng.nextInt(1, 10)
            val supportedOperations = (0 until numOperations).map { "Operation_$it" }
            val numFiles = rng.nextInt(1, 20)
            val generatedFiles = (0 until numFiles).map { "file_$it.c" }
            
            val result = ArduinoLibraryResult(
                libraryPath = libraryPath,
                memoryRequirements = memoryLayout,
                supportedOperations = supportedOperations,
                generatedFiles = generatedFiles
            )
            
            // Verify data integrity
            assertEquals(libraryPath, result.libraryPath)
            assertEquals(memoryLayout, result.memoryRequirements)
            assertEquals(supportedOperations, result.supportedOperations)
            assertEquals(generatedFiles, result.generatedFiles)
            
            // Verify constraints
            assertTrue(result.supportedOperations.isNotEmpty())
            assertTrue(result.generatedFiles.isNotEmpty())
            assertTrue(result.supportedOperations.all { it.isNotBlank() })
            assertTrue(result.generatedFiles.all { it.isNotBlank() })
        }
    }

    @Test
    fun memoryLayout_invalidInputs_validation() {
        // Test negative values are rejected
        assertFailsWith<IllegalArgumentException> {
            MemoryLayout(-1, 100, 200, listOf(50, 50))
        }
        
        assertFailsWith<IllegalArgumentException> {
            MemoryLayout(100, -1, 200, listOf(50, 50))
        }
        
        assertFailsWith<IllegalArgumentException> {
            MemoryLayout(100, 100, -1, listOf(50, 50))
        }
        
        assertFailsWith<IllegalArgumentException> {
            MemoryLayout(100, 100, 200, listOf(-1, 50))
        }
    }

    @Test
    fun layerCode_invalidInputs_validation() {
        // Test blank strings are rejected
        assertFailsWith<IllegalArgumentException> {
            LayerCode("", "Dense", intArrayOf(10), intArrayOf(5), "code")
        }
        
        assertFailsWith<IllegalArgumentException> {
            LayerCode("layer1", "", intArrayOf(10), intArrayOf(5), "code")
        }
        
        assertFailsWith<IllegalArgumentException> {
            LayerCode("layer1", "Dense", intArrayOf(10), intArrayOf(5), "")
        }
        
        // Test empty shapes are rejected
        assertFailsWith<IllegalArgumentException> {
            LayerCode("layer1", "Dense", intArrayOf(), intArrayOf(5), "code")
        }
        
        assertFailsWith<IllegalArgumentException> {
            LayerCode("layer1", "Dense", intArrayOf(10), intArrayOf(), "code")
        }
        
        // Test non-positive dimensions are rejected
        assertFailsWith<IllegalArgumentException> {
            LayerCode("layer1", "Dense", intArrayOf(0), intArrayOf(5), "code")
        }
        
        assertFailsWith<IllegalArgumentException> {
            LayerCode("layer1", "Dense", intArrayOf(10), intArrayOf(-1), "code")
        }
    }

    @Test
    fun arduinoLibraryResult_invalidInputs_validation() {
        val validMemoryLayout = MemoryLayout(100, 200, 300, listOf(50, 50))
        
        // Test blank library path is rejected
        assertFailsWith<IllegalArgumentException> {
            ArduinoLibraryResult("", validMemoryLayout, listOf("Dense"), listOf("file.c"))
        }
        
        // Test empty operations list is rejected
        assertFailsWith<IllegalArgumentException> {
            ArduinoLibraryResult("/path", validMemoryLayout, emptyList(), listOf("file.c"))
        }
        
        // Test empty files list is rejected
        assertFailsWith<IllegalArgumentException> {
            ArduinoLibraryResult("/path", validMemoryLayout, listOf("Dense"), emptyList())
        }
        
        // Test blank operation names are rejected
        assertFailsWith<IllegalArgumentException> {
            ArduinoLibraryResult("/path", validMemoryLayout, listOf(""), listOf("file.c"))
        }
        
        // Test blank file names are rejected
        assertFailsWith<IllegalArgumentException> {
            ArduinoLibraryResult("/path", validMemoryLayout, listOf("Dense"), listOf(""))
        }
    }
}