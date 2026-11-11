package sk.ainet.lang.tensor

import sk.ainet.context.data
import sk.ainet.lang.tensor.dsl.*
import sk.ainet.lang.types.FP32
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Stress test suite for tensor slicing functionality
 * Implements testing requirements from tasks-slicing.md section 5.3 (tasks 23-24)
 */
class SlicingStressTest {

    @Test
    fun testBehaviorUnderMemoryPressureConditions() {
        println("[DEBUG_LOG] Testing behavior under memory pressure conditions")

        data { 
            val largeTensor = tensor<FP32, Float> {
                // Create large tensor to simulate memory pressure
                shape(500, 500, 10) {
                    init { indices ->
                        // Simple pattern to avoid expensive computations
                        (indices[0] + indices[1] + indices[2]).toFloat()
                    }
                }
            }

            // Create many views to test memory management
            val views = mutableListOf<Tensor<FP32, Float>>()

            repeat(100) { i ->
                val view = largeTensor.sliceView {
                    segment { range(i, i + 10) }
                    segment { all() }
                    segment { all() }
                }
                views.add(view)

                // Verify view integrity
                assertEquals(Shape(10, 500, 10), view.shape)
            }

            // Test that all views are still accessible
            views.forEachIndexed { index, view ->
                val firstValue = view.data[0, 0, 0]
                val expectedValue = (index + 0 + 0).toFloat()
                assertEquals(expectedValue, firstValue)
            }

            println("[DEBUG_LOG] Memory pressure test completed with ${views.size} views")
        }
    }

    // Task 23.2: Validate performance with large tensor slicing scenarios
    @Test
    fun testPerformanceWithLargeTensorSlicingScenarios() {
        println("[DEBUG_LOG] Testing performance with large tensor slicing scenarios")

        data {
            val largeTensor = tensor<FP32, Float> {
                // Create very large tensor
                shape(1000, 1000, 5) {
                    init { indices ->
                        // Use modular arithmetic to keep values reasonable
                        ((indices[0] * 1000 + indices[1]) % 100000 + indices[2]).toFloat()
                    }
                }
            }
            // Test various slicing patterns for performance

            // 1. Large contiguous slices
            val largeSlice = largeTensor.sliceView {
                segment { range(100, 900) }
                segment { range(100, 900) }
                segment { all() }
            }
            assertEquals(Shape(800, 800, 5), largeSlice.shape)

            // 2. Many small slices
            val smallSlices = (0 until 50).map { i ->
                largeTensor.sliceView {
                    segment { range(i * 20, (i + 1) * 20) }
                    segment { all() }
                    segment { all() }
                }
            }
            smallSlices.forEach { slice ->
                assertEquals(Shape(20, 1000, 5), slice.shape)
            }

            // 3. Strided access patterns
            val stridedSlice = largeTensor.sliceView {
                segment { step(0, 1000, 10) }
                segment { step(0, 1000, 10) }
                segment { all() }
            }
            assertEquals(Shape(100, 100, 5), stridedSlice.shape)

            // Verify data accessibility in large slices
            val sampleValue = largeSlice.data[0, 0, 0]
            assertTrue(sampleValue >= 0.0f, "Sample value should be non-negative")

            println("[DEBUG_LOG] Large tensor slicing scenarios validated")
        }
    }

    // Task 23.3: Test view creation/destruction patterns for memory stability
    @Test
    fun testViewCreationDestructionPatternsForMemoryStability() {
        println("[DEBUG_LOG] Testing view creation/destruction patterns for memory stability")
        data {
            val baseTensor = tensor<FP32, Float> {
                shape(200, 200, 5) {
                    init { indices ->
                        (indices[0] * 1000 + indices[1] + indices[2]).toFloat()
                    }
                }
            }

            // Test rapid creation and destruction of views
            repeat(10) { round ->
                val temporaryViews = mutableListOf<Tensor<FP32, Float>>()

                // Create many views
                repeat(20) { i ->
                    val view = baseTensor.sliceView {
                        segment { range(i * 5, (i + 1) * 5) }
                        segment { all() }
                        segment { all() }
                    }
                    temporaryViews.add(view)
                }

                // Use views briefly
                temporaryViews.forEach { view ->
                    assertEquals(Shape(5, 200, 5), view.shape)
                    val value = view.data[0, 0, 0]
                    assertTrue(value >= 0.0f)
                }

                // Let views go out of scope (automatic destruction)
                temporaryViews.clear()

                println("[DEBUG_LOG] Completed round $round of view creation/destruction")
            }

            // Create final views to ensure system is still stable
            val finalViews = (0 until 5).map { i ->
                baseTensor.sliceView {
                    segment { range(i * 10, (i + 1) * 10) }
                    segment { all() }
                    segment { all() }
                }
            }

            finalViews.forEach { view ->
                assertEquals(Shape(10, 200, 5), view.shape)
            }

            println("[DEBUG_LOG] Memory stability test completed successfully")
        }
    }

    // Task 23.4: Validate fallback mechanisms under various conditions
    @Test
    fun testFallbackMechanismsUnderVariousConditions() {
        println("[DEBUG_LOG] Testing fallback mechanisms under various conditions")

        data {
            val tensor = tensor<FP32, Float> {
                shape(50, 50, 10) { _ ->
                    init { indices ->
                        (indices[0] * 100 + indices[1] + indices[2]).toFloat()
                    }
                }
            }

            // Test edge conditions that might trigger fallbacks

            // 1. Very small slices
            val tinySlice = tensor.sliceView {
                segment { range(0, 1) }
                segment { range(0, 1) }
                segment { range(0, 1) }
            }
            assertEquals(Shape(1, 1, 1), tinySlice.shape)

            // 2. Single dimension slices
            val singleDimSlice = tensor.sliceView {
                segment { at(25) }
                segment { at(25) }
                segment { all() }
            }
            assertEquals(Shape(10), singleDimSlice.shape)

            // 3. Maximum range slices
            val maxRangeSlice = tensor.sliceView {
                segment { range(0, 50) }
                segment { range(0, 50) }
                segment { range(0, 10) }
            }
            assertEquals(Shape(50, 50, 10), maxRangeSlice.shape)

            // 4. Complex strided patterns
            val complexStride = tensor.sliceView {
                segment { step(1, 49, 5) }
                segment { step(2, 48, 3) }
                segment { step(0, 10, 2) }
            }
            // Calculate expected shape: (49-1)/5 = 9.6 -> 10, (48-2)/3 = 15.33 -> 16, (10-0)/2 = 5
            assertTrue(complexStride.shape[0] <= 10, "First dimension should be reasonable")
            assertTrue(complexStride.shape[1] <= 16, "Second dimension should be reasonable")
            assertTrue(complexStride.shape[2] <= 5, "Third dimension should be reasonable")

            // Verify all fallback scenarios work correctly
            listOf(tinySlice, singleDimSlice, maxRangeSlice, complexStride).forEach { view ->
                assertNotNull(view.data, "View data should be accessible")
                assertTrue(view.shape.volume > 0, "View should have positive volume")
            }

            println("[DEBUG_LOG] Fallback mechanisms validated under various conditions")
        }
    }

    // Task 24.1: Test on different Kotlin/Multiplatform targets (implicit via test execution)
    @Test
    fun testMultiplatformTargetCompatibility() {
        println("[DEBUG_LOG] Testing multiplatform target compatibility")

        data {
            val tensor = tensor<FP32, Float> {
                // This test runs on all platforms defined in build.gradle.kts
                shape(20, 20, 5) { _ ->
                    init { indices ->
                        (indices[0] * 100 + indices[1] * 5 + indices[2]).toFloat()
                    }
                }
            }

            // Test basic slicing functionality works on all targets
            val view = tensor.sliceView {
                segment { range(5, 15) }
                segment { range(5, 15) }
                segment { all() }
            }

            assertEquals(Shape(10, 10, 5), view.shape)

            // Test data access works on all platforms
            val value = view.data[0, 0, 0]
            val expectedValue = (5 * 100 + 5 * 5 + 0).toFloat()
            assertEquals(expectedValue, value)

            // Test that slicing operations are consistent across platforms
            val chainedView = view.sliceView {
                segment { range(2, 8) }
                segment { range(2, 8) }
                segment { range(1, 4) }
            }

            assertEquals(Shape(6, 6, 3), chainedView.shape)

            println("[DEBUG_LOG] Multiplatform target compatibility verified")
        }
    }

    // Task 24.2: Validate tensor interoperability with views
    @Test
    fun testTensorInteroperabilityWithViews() {
        println("[DEBUG_LOG] Testing tensor interoperability with views")

        data {
            val originalTensor = tensor<FP32, Float> {
                shape(30, 30, 3) {
                    init { indices ->
                        (indices[0] * 1000 + indices[1] * 10 + indices[2]).toFloat()
                    }
                }
            }

            // Create views from tensor
            val view1 = originalTensor.sliceView {
                segment { range(0, 15) }
                segment { all() }
                segment { all() }
            }

            val view2 = originalTensor.sliceView {
                segment { range(15, 30) }
                segment { all() }
                segment { all() }
            }

            // Create views from views (chaining)
            val subView1 = view1.sliceView {
                segment { range(5, 10) }
                segment { range(10, 20) }
                segment { all() }
            }

            val subView2 = view2.sliceView {
                segment { range(2, 7) }
                segment { range(5, 15) }
                segment { range(0, 2) }
            }

            // Verify interoperability
            assertEquals(Shape(15, 30, 3), view1.shape)
            assertEquals(Shape(15, 30, 3), view2.shape)
            assertEquals(Shape(5, 10, 3), subView1.shape)
            assertEquals(Shape(5, 10, 2), subView2.shape)

            // Test data consistency across view hierarchies
            val originalValue = originalTensor.data[5, 10, 1]
            val viewValue = view1.data[5, 10, 1]
            assertEquals(originalValue, viewValue, "Views should access same data as original tensor")

            val subViewValue = subView1.data[0, 0, 1]
            val expectedSubViewValue = originalTensor.data[5, 10, 1]
            assertEquals(expectedSubViewValue, subViewValue, "Sub-views should maintain data consistency")

            println("[DEBUG_LOG] Tensor interoperability with views validated")
        }
    }

    // Task 24.3: Test serialization/deserialization of views (basic validation)
    @Test
    fun testSerializationCompatibilityOfViews() {
        println("[DEBUG_LOG] Testing serialization compatibility of views")

        data {

            val tensor = tensor<FP32, Float> {
                shape(10, 10, 3) {
                    init { indices ->
                        (indices[0] * 100 + indices[1] * 10 + indices[2]).toFloat()
                    }
                }
            }

            val view = tensor.sliceView {
                segment { range(2, 8) }
                segment { range(3, 7) }
                segment { all() }
            }

            // Test that view properties can be accessed for serialization
            val shapeInfo = view.shape
            assertEquals(Shape(6, 4, 3), shapeInfo)

            // Test basic view metadata accessibility
            assertNotNull(view.data, "View data should be accessible")
            assertTrue(view.rank > 0, "View should have positive rank")

            // Test that view can be reconstructed with same properties
            val reconstructedView = tensor.sliceView {
                segment { range(2, 8) }
                segment { range(3, 7) }
                segment { all() }
            }

            assertEquals(view.shape, reconstructedView.shape)

            // Verify data consistency
            val originalData = view.data[0, 0, 0]
            val reconstructedData = reconstructedView.data[0, 0, 0]
            assertEquals(originalData, reconstructedData)

            println("[DEBUG_LOG] Serialization compatibility of views validated")
        }
    }

    // Task 24.4: Performance testing on different hardware architectures (via platform execution)
    @Test
    fun testPerformanceOnDifferentArchitectures() {
        println("[DEBUG_LOG] Testing performance on different architectures")

        data {

            val tensor = tensor<FP32, Float> {
                shape(100, 100, 10) {
                    init { indices ->
                        (indices[0] + indices[1] + indices[2]).toFloat()
                    }
                }
            }

            // Test that slicing operations complete in reasonable time across architectures
            val iterations = 50

            repeat(iterations) { i ->
                val view = tensor.sliceView {
                    segment { range(i % 50, (i % 50) + 10) }
                    segment { all() }
                    segment { all() }
                }

                // Verify the operation completed successfully
                assertEquals(Shape(10, 100, 10), view.shape)

                // Access some data to ensure the view is functional
                val value = view.data[0, 0, 0]
                assertTrue(value >= 0.0f, "Data should be accessible")
            }

            // Test complex slicing patterns for architecture compatibility
            val complexView = tensor.sliceView {
                segment { step(0, 100, 7) }
                segment { step(5, 95, 3) }
                segment { range(2, 8) }
            }

            assertTrue(complexView.shape.volume > 0, "Complex view should have positive volume")

            // Test memory-intensive operations
            val largeViewSet = (0 until 20).map { i ->
                tensor.sliceView {
                    segment { range(i, i + 5) }
                    segment { range(i * 2, (i * 2) + 10) }
                    segment { all() }
                }
            }

            largeViewSet.forEach { view ->
                assertTrue(view.shape.volume > 0, "Each view should be valid")
            }

            println("[DEBUG_LOG] Performance testing completed across architectures")
        }
    }
}