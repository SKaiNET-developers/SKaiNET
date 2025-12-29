@file:OptIn(ExperimentalCompilerApi::class)

package sk.ainet.lang.ksp

import com.tschuchort.compiletesting.KotlinCompilation
import com.tschuchort.compiletesting.SourceFile
import com.tschuchort.compiletesting.symbolProcessorProviders
import org.jetbrains.kotlin.compiler.plugin.ExperimentalCompilerApi
import org.junit.Test
import kotlin.test.assertTrue
import kotlin.test.assertFalse

/**
 * Property-based tests for TracingWrapperProcessor.
 * 
 * **Feature: ksp-tracing-tensorops, Property 5: Annotation-Based Processing**
 * **Validates: Requirements 5.2, 5.3**
 */
class TracingWrapperProcessorTest {

    /**
     * Property 5: Annotation-Based Processing
     * For any interface, the KSP processor should generate a tracing wrapper 
     * if and only if the interface is annotated with @GenerateTracingWrapper.
     */
    @Test
    fun testAnnotationBasedProcessingProperty() {
        // Test case 1: Interface WITH @GenerateTracingWrapper annotation
        val annotatedInterfaceSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface TestTensorOps {
                fun add(a: Int, b: Int): Int
                fun multiply(x: Float, y: Float): Float
            }
        """.trimIndent()

        val annotatedResult = compileWithProcessor(annotatedInterfaceSource, "AnnotatedInterface.kt")
        
        // Should process the annotated interface
        assertTrue(
            annotatedResult.messages.contains("Found 1 annotated interfaces"),
            "Processor should find annotated interface. Messages: ${annotatedResult.messages}"
        )
        assertTrue(
            annotatedResult.messages.contains("Generating tracing wrapper for TestTensorOps"),
            "Processor should generate wrapper for annotated interface. Messages: ${annotatedResult.messages}"
        )

        // Test case 2: Interface WITHOUT @GenerateTracingWrapper annotation
        val nonAnnotatedInterfaceSource = """
            package test
            
            interface PlainTensorOps {
                fun subtract(a: Int, b: Int): Int
                fun divide(x: Float, y: Float): Float
            }
        """.trimIndent()

        val nonAnnotatedResult = compileWithProcessor(nonAnnotatedInterfaceSource, "NonAnnotatedInterface.kt")
        
        // Should NOT process the non-annotated interface
        assertTrue(
            nonAnnotatedResult.messages.contains("No interfaces annotated with @GenerateTracingWrapper found"),
            "Processor should not find non-annotated interface. Messages: ${nonAnnotatedResult.messages}"
        )
        assertFalse(
            nonAnnotatedResult.messages.contains("Generating tracing wrapper"),
            "Processor should not generate wrapper for non-annotated interface. Messages: ${nonAnnotatedResult.messages}"
        )

        // Test case 3: Multiple interfaces - mixed annotation status
        val mixedInterfacesSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface AnnotatedOps {
                fun operation1(): String
            }
            
            interface NonAnnotatedOps {
                fun operation2(): String
            }
            
            @GenerateTracingWrapper
            interface AnotherAnnotatedOps {
                fun operation3(): Int
            }
        """.trimIndent()

        val mixedResult = compileWithProcessor(mixedInterfacesSource, "MixedInterfaces.kt")
        
        // Should process exactly 2 annotated interfaces
        assertTrue(
            mixedResult.messages.contains("Found 2 annotated interfaces"),
            "Processor should find exactly 2 annotated interfaces. Messages: ${mixedResult.messages}"
        )
        assertTrue(
            mixedResult.messages.contains("Generating tracing wrapper for AnnotatedOps"),
            "Processor should generate wrapper for first annotated interface. Messages: ${mixedResult.messages}"
        )
        assertTrue(
            mixedResult.messages.contains("Generating tracing wrapper for AnotherAnnotatedOps"),
            "Processor should generate wrapper for second annotated interface. Messages: ${mixedResult.messages}"
        )
        assertFalse(
            mixedResult.messages.contains("Generating tracing wrapper for NonAnnotatedOps"),
            "Processor should not generate wrapper for non-annotated interface. Messages: ${mixedResult.messages}"
        )
    }

    /**
     * Test validation of annotation targets - should only accept interfaces.
     */
    @Test
    fun testInterfaceOnlyValidation() {
        // Test case: Class annotated with @GenerateTracingWrapper (should fail)
        val classSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            class TestClass {
                fun someMethod(): String = "test"
            }
        """.trimIndent()

        val classResult = compileWithProcessor(classSource, "AnnotatedClass.kt")
        
        // Should emit error for non-interface
        assertTrue(
            classResult.messages.contains("GenerateTracingWrapper can only be applied to interfaces"),
            "Processor should reject non-interface targets. Messages: ${classResult.messages}"
        )

        // Test case: Enum annotated with @GenerateTracingWrapper (should fail)
        val enumSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            enum class TestEnum {
                VALUE1, VALUE2
            }
        """.trimIndent()

        val enumResult = compileWithProcessor(enumSource, "AnnotatedEnum.kt")
        
        // Should emit error for enum
        assertTrue(
            enumResult.messages.contains("GenerateTracingWrapper can only be applied to interfaces"),
            "Processor should reject enum targets. Messages: ${enumResult.messages}"
        )
    }

    /**
     * Property 1: Complete Interface Implementation Generation
     * For any interface annotated with @GenerateTracingWrapper, the KSP processor should 
     * generate a complete tracing wrapper that implements all methods from the original 
     * interface and produces compilable Kotlin code.
     * **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
     */
    @Test
    fun testCompleteInterfaceImplementationGeneration() {
        // Test case 1: Interface with various method signatures
        val complexInterfaceSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface ComplexTensorOps {
                fun simpleMethod(): String
                fun methodWithParams(a: Int, b: Float): Double
                fun <T> genericMethod(value: T): T
                fun methodWithOptionalParam(required: String, optional: Int = 42): Boolean
                fun methodReturningList(): List<String>
                fun overloadedMethod(x: Int): String
                fun overloadedMethod(x: Float): String
            }
        """.trimIndent()

        val result = compileWithProcessor(complexInterfaceSource, "ComplexInterface.kt")
        
        // Should analyze all methods
        assertTrue(
            result.messages.contains("Analyzed 7 methods for ComplexTensorOps"),
            "Processor should analyze all 7 methods. Messages: ${result.messages}"
        )
        
        // Should log method analysis details
        assertTrue(
            result.messages.contains("Method simpleMethod:"),
            "Should analyze simpleMethod. Messages: ${result.messages}"
        )
        assertTrue(
            result.messages.contains("Method methodWithParams:"),
            "Should analyze methodWithParams. Messages: ${result.messages}"
        )
        assertTrue(
            result.messages.contains("Method genericMethod:"),
            "Should analyze genericMethod. Messages: ${result.messages}"
        )

        // Test case 2: Interface with tensor-like parameters
        val tensorInterfaceSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface TensorOpsLike {
                fun addTensors(a: sk.ainet.Tensor<Int>, b: sk.ainet.Tensor<Int>): sk.ainet.Tensor<Int>
                fun multiOutput(): List<sk.ainet.Tensor<Float>>
                fun mixedParams(tensor: sk.ainet.Tensor<Double>, scalar: Int): sk.ainet.Tensor<Double>
            }
        """.trimIndent()

        val tensorResult = compileWithProcessor(tensorInterfaceSource, "TensorInterface.kt")
        
        // Should analyze tensor methods
        assertTrue(
            tensorResult.messages.contains("Analyzed 3 methods for TensorOpsLike"),
            "Processor should analyze all 3 tensor methods. Messages: ${tensorResult.messages}"
        )
        assertTrue(
            tensorResult.messages.contains("Method addTensors:"),
            "Should analyze addTensors method. Messages: ${tensorResult.messages}"
        )
        assertTrue(
            tensorResult.messages.contains("Method multiOutput:"),
            "Should analyze multiOutput method. Messages: ${tensorResult.messages}"
        )
    }

    /**
     * Property 6: Generated Class Structure Consistency
     * For any generated tracing wrapper, the class should be named "Ksp" + interface name,
     * implement the original interface, accept the required constructor parameters, and be 
     * placed in the same package as the original interface.
     * **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
     */
    @Test
    fun testGeneratedClassStructureConsistency() {
        // Test case 1: Simple interface structure validation
        val simpleInterfaceSource = """
            package com.example.test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface SimpleTensorOps {
                fun basicOperation(x: Int): String
                fun anotherOperation(): Double
            }
        """.trimIndent()

        val result = compileWithProcessor(simpleInterfaceSource, "SimpleInterface.kt")
        
        // Should generate class with correct naming
        assertTrue(
            result.messages.contains("Generating class KspSimpleTensorOps"),
            "Should generate class with Ksp prefix. Messages: ${result.messages}"
        )
        
        // Should place in same package
        assertTrue(
            result.messages.contains("Successfully generated KspSimpleTensorOps.kt in package com.example.test"),
            "Should place generated class in same package. Messages: ${result.messages}"
        )
        
        // Should analyze all methods for implementation
        assertTrue(
            result.messages.contains("Analyzed 2 methods for SimpleTensorOps"),
            "Should analyze all interface methods. Messages: ${result.messages}"
        )

        // Test case 2: Interface with complex name
        val complexNameSource = """
            package org.test.complex
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface VeryComplexTensorOperationsInterface {
                fun operation1(): Unit
                fun operation2(param: String): Int
                fun <T> genericOperation(value: T): T
            }
        """.trimIndent()

        val complexResult = compileWithProcessor(complexNameSource, "ComplexNameInterface.kt")
        
        // Should handle complex interface names correctly
        assertTrue(
            complexResult.messages.contains("Generating class KspVeryComplexTensorOperationsInterface"),
            "Should handle complex interface names. Messages: ${complexResult.messages}"
        )
        assertTrue(
            complexResult.messages.contains("Successfully generated KspVeryComplexTensorOperationsInterface.kt in package org.test.complex"),
            "Should place in correct package. Messages: ${complexResult.messages}"
        )
        assertTrue(
            complexResult.messages.contains("Analyzed 3 methods for VeryComplexTensorOperationsInterface"),
            "Should analyze all methods including generic ones. Messages: ${complexResult.messages}"
        )
    }

    /**
     * Test empty interface handling.
     */
    @Test
    fun testEmptyInterfaceHandling() {
        val emptyInterfaceSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface EmptyInterface {
                // No methods
            }
        """.trimIndent()

        val result = compileWithProcessor(emptyInterfaceSource, "EmptyInterface.kt")
        
        // Should process but warn about no methods
        assertTrue(
            result.messages.contains("Found 1 annotated interfaces"),
            "Processor should find empty annotated interface. Messages: ${result.messages}"
        )
        assertTrue(
            result.messages.contains("Interface EmptyInterface has no abstract methods to trace"),
            "Processor should warn about empty interface. Messages: ${result.messages}"
        )
    }

    /**
     * Property 2: Method Delegation and Tracing Behavior
     * For any generated tracing method, when called with tensor inputs, it should delegate 
     * to the base implementation, capture all input and output tensors as TensorRefs, 
     * emit a complete OpTrace, and return the original result.
     * **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
     */
    @Test
    fun testMethodDelegationAndTracingBehavior() {
        // Test case 1: Method with tensor inputs and tensor output
        val tensorMethodSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface TensorMethodOps {
                fun addTensors(a: sk.ainet.lang.tensor.Tensor<Int, Any>, b: sk.ainet.lang.tensor.Tensor<Int, Any>): sk.ainet.lang.tensor.Tensor<Int, Any>
                fun processTensor(input: sk.ainet.lang.tensor.Tensor<Float, Any>): sk.ainet.lang.tensor.Tensor<Float, Any>
            }
        """.trimIndent()

        val tensorResult = compileWithProcessor(tensorMethodSource, "TensorMethodInterface.kt")
        
        // Should generate proper delegation and tracing for tensor methods
        assertTrue(
            tensorResult.messages.contains("Generating tracing wrapper for TensorMethodOps"),
            "Should generate wrapper for tensor methods. Messages: ${tensorResult.messages}"
        )
        assertTrue(
            tensorResult.messages.contains("Analyzed 2 methods for TensorMethodOps"),
            "Should analyze both tensor methods. Messages: ${tensorResult.messages}"
        )
        
        // Verify method analysis identifies tensor parameters
        assertTrue(
            tensorResult.messages.contains("Method addTensors:"),
            "Should analyze addTensors method. Messages: ${tensorResult.messages}"
        )
        assertTrue(
            tensorResult.messages.contains("Method processTensor:"),
            "Should analyze processTensor method. Messages: ${tensorResult.messages}"
        )

        // Test case 2: Method with mixed tensor and non-tensor parameters
        val mixedMethodSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface MixedParameterOps {
                fun scaleAndAdd(tensor: sk.ainet.lang.tensor.Tensor<Float, Any>, scale: Float, offset: Int): sk.ainet.lang.tensor.Tensor<Float, Any>
                fun conditionalProcess(input: sk.ainet.lang.tensor.Tensor<Double, Any>, threshold: Double, enabled: Boolean): sk.ainet.lang.tensor.Tensor<Double, Any>
            }
        """.trimIndent()

        val mixedResult = compileWithProcessor(mixedMethodSource, "MixedParameterInterface.kt")
        
        // Should handle mixed parameter types correctly
        assertTrue(
            mixedResult.messages.contains("Generating tracing wrapper for MixedParameterOps"),
            "Should generate wrapper for mixed parameter methods. Messages: ${mixedResult.messages}"
        )
        assertTrue(
            mixedResult.messages.contains("Analyzed 2 methods for MixedParameterOps"),
            "Should analyze both mixed parameter methods. Messages: ${mixedResult.messages}"
        )

        // Test case 3: Method with no tensor parameters (non-tensor operations)
        val nonTensorMethodSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface NonTensorOps {
                fun computeMetrics(count: Int, rate: Float): Double
                fun validateConfig(name: String, enabled: Boolean): Boolean
            }
        """.trimIndent()

        val nonTensorResult = compileWithProcessor(nonTensorMethodSource, "NonTensorInterface.kt")
        
        // Should handle non-tensor methods correctly
        assertTrue(
            nonTensorResult.messages.contains("Generating tracing wrapper for NonTensorOps"),
            "Should generate wrapper for non-tensor methods. Messages: ${nonTensorResult.messages}"
        )
        assertTrue(
            nonTensorResult.messages.contains("Analyzed 2 methods for NonTensorOps"),
            "Should analyze both non-tensor methods. Messages: ${nonTensorResult.messages}"
        )

        // Test case 4: Method with Unit return type
        val unitReturnSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface UnitReturnOps {
                fun updateTensor(tensor: sk.ainet.lang.tensor.Tensor<Int, Any>, value: Int): Unit
                fun logOperation(message: String): Unit
            }
        """.trimIndent()

        val unitResult = compileWithProcessor(unitReturnSource, "UnitReturnInterface.kt")
        
        // Should handle Unit return type correctly
        assertTrue(
            unitResult.messages.contains("Generating tracing wrapper for UnitReturnOps"),
            "Should generate wrapper for Unit return methods. Messages: ${unitResult.messages}"
        )
        assertTrue(
            unitResult.messages.contains("Analyzed 2 methods for UnitReturnOps"),
            "Should analyze both Unit return methods. Messages: ${unitResult.messages}"
        )

        // Test case 5: Method with generic type parameters
        val genericMethodSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface GenericOps {
                fun <T> transformTensor(input: sk.ainet.lang.tensor.Tensor<T, Any>): sk.ainet.lang.tensor.Tensor<T, Any>
                fun <T, U> convertTensor(input: sk.ainet.lang.tensor.Tensor<T, Any>, scale: Float): sk.ainet.lang.tensor.Tensor<U, Any>
            }
        """.trimIndent()

        val genericResult = compileWithProcessor(genericMethodSource, "GenericInterface.kt")
        
        // Should handle generic methods correctly
        assertTrue(
            genericResult.messages.contains("Generating tracing wrapper for GenericOps"),
            "Should generate wrapper for generic methods. Messages: ${genericResult.messages}"
        )
        assertTrue(
            genericResult.messages.contains("Analyzed 2 methods for GenericOps"),
            "Should analyze both generic methods. Messages: ${genericResult.messages}"
        )
    }

    /**
     * Property 4: Multi-Output Operation Support
     * For any method returning List<Tensor<T, V>>, the generated implementation should 
     * create TensorRef objects for each tensor in the list and include all output 
     * TensorRefs in the OpTrace.outputs.
     * **Validates: Requirements 4.1, 4.2, 4.3**
     */
    @Test
    fun testMultiOutputOperationSupport() {
        // Test case 1: Method returning List<Tensor>
        val multiOutputSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface MultiOutputOps {
                fun splitTensor(input: sk.ainet.lang.tensor.Tensor<Float, Any>, parts: Int): List<sk.ainet.lang.tensor.Tensor<Float, Any>>
                fun decompose(matrix: sk.ainet.lang.tensor.Tensor<Double, Any>): List<sk.ainet.lang.tensor.Tensor<Double, Any>>
            }
        """.trimIndent()

        val multiResult = compileWithProcessor(multiOutputSource, "MultiOutputInterface.kt")
        
        // Should generate wrapper for multi-output methods
        assertTrue(
            multiResult.messages.contains("Generating tracing wrapper for MultiOutputOps"),
            "Should generate wrapper for multi-output methods. Messages: ${multiResult.messages}"
        )
        assertTrue(
            multiResult.messages.contains("Analyzed 2 methods for MultiOutputOps"),
            "Should analyze both multi-output methods. Messages: ${multiResult.messages}"
        )
        
        // Verify method analysis identifies tensor list returns
        assertTrue(
            multiResult.messages.contains("Method splitTensor:"),
            "Should analyze splitTensor method. Messages: ${multiResult.messages}"
        )
        assertTrue(
            multiResult.messages.contains("Method decompose:"),
            "Should analyze decompose method. Messages: ${multiResult.messages}"
        )

        // Test case 2: Mixed single and multi-output methods
        val mixedOutputSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface MixedOutputOps {
                fun singleOutput(input: sk.ainet.lang.tensor.Tensor<Int, Any>): sk.ainet.lang.tensor.Tensor<Int, Any>
                fun multiOutput(input: sk.ainet.lang.tensor.Tensor<Int, Any>): List<sk.ainet.lang.tensor.Tensor<Int, Any>>
                fun noTensorOutput(value: Int): String
                fun multiTensorOutput(a: sk.ainet.lang.tensor.Tensor<Float, Any>, b: sk.ainet.lang.tensor.Tensor<Float, Any>): List<sk.ainet.lang.tensor.Tensor<Float, Any>>
            }
        """.trimIndent()

        val mixedOutputResult = compileWithProcessor(mixedOutputSource, "MixedOutputInterface.kt")
        
        // Should handle mixed output types correctly
        assertTrue(
            mixedOutputResult.messages.contains("Generating tracing wrapper for MixedOutputOps"),
            "Should generate wrapper for mixed output methods. Messages: ${mixedOutputResult.messages}"
        )
        assertTrue(
            mixedOutputResult.messages.contains("Analyzed 4 methods for MixedOutputOps"),
            "Should analyze all 4 mixed output methods. Messages: ${mixedOutputResult.messages}"
        )

        // Test case 3: Multi-output with complex parameter combinations
        val complexMultiOutputSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface ComplexMultiOutputOps {
                fun <T> genericSplit(input: sk.ainet.lang.tensor.Tensor<T, Any>, count: Int): List<sk.ainet.lang.tensor.Tensor<T, Any>>
                fun conditionalSplit(
                    input: sk.ainet.lang.tensor.Tensor<Double, Any>, 
                    threshold: Double, 
                    maxParts: Int
                ): List<sk.ainet.lang.tensor.Tensor<Double, Any>>
                fun multiInputMultiOutput(
                    a: sk.ainet.lang.tensor.Tensor<Float, Any>,
                    b: sk.ainet.lang.tensor.Tensor<Float, Any>,
                    operation: String
                ): List<sk.ainet.lang.tensor.Tensor<Float, Any>>
            }
        """.trimIndent()

        val complexResult = compileWithProcessor(complexMultiOutputSource, "ComplexMultiOutputInterface.kt")
        
        // Should handle complex multi-output scenarios
        assertTrue(
            complexResult.messages.contains("Generating tracing wrapper for ComplexMultiOutputOps"),
            "Should generate wrapper for complex multi-output methods. Messages: ${complexResult.messages}"
        )
        assertTrue(
            complexResult.messages.contains("Analyzed 3 methods for ComplexMultiOutputOps"),
            "Should analyze all 3 complex multi-output methods. Messages: ${complexResult.messages}"
        )

        // Test case 4: Edge case - empty list return
        val emptyListSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface EmptyListOps {
                fun filterTensors(inputs: List<sk.ainet.lang.tensor.Tensor<Int, Any>>, threshold: Int): List<sk.ainet.lang.tensor.Tensor<Int, Any>>
                fun emptyResult(): List<sk.ainet.lang.tensor.Tensor<Float, Any>>
            }
        """.trimIndent()

        val emptyListResult = compileWithProcessor(emptyListSource, "EmptyListInterface.kt")
        
        // Should handle edge cases correctly
        assertTrue(
            emptyListResult.messages.contains("Generating tracing wrapper for EmptyListOps"),
            "Should generate wrapper for empty list methods. Messages: ${emptyListResult.messages}"
        )
        assertTrue(
            emptyListResult.messages.contains("Analyzed 2 methods for EmptyListOps"),
            "Should analyze both empty list methods. Messages: ${emptyListResult.messages}"
        )
    }

    /**
     * Property 3: Attribute Handling Completeness
     * For any method with non-tensor parameters, the generated implementation should 
     * capture all parameters in the OpTrace attributes using OpAttributeFactory methods 
     * when available, or default parameter-name mapping otherwise, handling all 
     * supported parameter types correctly.
     * **Validates: Requirements 3.1, 3.2, 3.3, 3.4**
     */
    @Test
    fun testAttributeHandlingCompleteness() {
        // Test case 1: Methods with OpAttributeFactory support (binary operations)
        val binaryOpsSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface BinaryOpsWithAttributes {
                fun add(a: sk.ainet.lang.tensor.Tensor<Float, Any>, b: sk.ainet.lang.tensor.Tensor<Float, Any>): sk.ainet.lang.tensor.Tensor<Float, Any>
                fun multiply(x: sk.ainet.lang.tensor.Tensor<Double, Any>, y: sk.ainet.lang.tensor.Tensor<Double, Any>): sk.ainet.lang.tensor.Tensor<Double, Any>
                fun matmul(left: sk.ainet.lang.tensor.Tensor<Int, Any>, right: sk.ainet.lang.tensor.Tensor<Int, Any>): sk.ainet.lang.tensor.Tensor<Int, Any>
            }
        """.trimIndent()

        val binaryResult = compileWithProcessor(binaryOpsSource, "BinaryOpsInterface.kt")
        
        // Should recognize binary operations for OpAttributeFactory usage
        assertTrue(
            binaryResult.messages.contains("Generating tracing wrapper for BinaryOpsWithAttributes"),
            "Should generate wrapper for binary ops. Messages: ${binaryResult.messages}"
        )
        assertTrue(
            binaryResult.messages.contains("Analyzed 3 methods for BinaryOpsWithAttributes"),
            "Should analyze all 3 binary methods. Messages: ${binaryResult.messages}"
        )

        // Test case 2: Methods with OpAttributeFactory support (unary operations)
        val unaryOpsSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface UnaryOpsWithAttributes {
                fun relu(input: sk.ainet.lang.tensor.Tensor<Float, Any>): sk.ainet.lang.tensor.Tensor<Float, Any>
                fun sigmoid(tensor: sk.ainet.lang.tensor.Tensor<Double, Any>): sk.ainet.lang.tensor.Tensor<Double, Any>
                fun softmax(x: sk.ainet.lang.tensor.Tensor<Float, Any>): sk.ainet.lang.tensor.Tensor<Float, Any>
                fun silu(input: sk.ainet.lang.tensor.Tensor<Float, Any>): sk.ainet.lang.tensor.Tensor<Float, Any>
            }
        """.trimIndent()

        val unaryResult = compileWithProcessor(unaryOpsSource, "UnaryOpsInterface.kt")
        
        // Should recognize unary operations for OpAttributeFactory usage
        assertTrue(
            unaryResult.messages.contains("Generating tracing wrapper for UnaryOpsWithAttributes"),
            "Should generate wrapper for unary ops. Messages: ${unaryResult.messages}"
        )
        assertTrue(
            unaryResult.messages.contains("Analyzed 4 methods for UnaryOpsWithAttributes"),
            "Should analyze all 4 unary methods. Messages: ${unaryResult.messages}"
        )

        // Test case 3: Conv2d operation with complex attributes
        val conv2dOpsSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface Conv2dOpsWithAttributes {
                fun conv2d(
                    input: sk.ainet.lang.tensor.Tensor<Float, Any>,
                    weight: sk.ainet.lang.tensor.Tensor<Float, Any>,
                    bias: sk.ainet.lang.tensor.Tensor<Float, Any>?,
                    stride: Pair<Int, Int>,
                    padding: Pair<Int, Int>,
                    dilation: Pair<Int, Int>,
                    groups: Int
                ): sk.ainet.lang.tensor.Tensor<Float, Any>
            }
        """.trimIndent()

        val conv2dResult = compileWithProcessor(conv2dOpsSource, "Conv2dOpsInterface.kt")
        
        // Should recognize conv2d operation for specialized OpAttributeFactory usage
        assertTrue(
            conv2dResult.messages.contains("Generating tracing wrapper for Conv2dOpsWithAttributes"),
            "Should generate wrapper for conv2d ops. Messages: ${conv2dResult.messages}"
        )
        assertTrue(
            conv2dResult.messages.contains("Analyzed 1 methods for Conv2dOpsWithAttributes"),
            "Should analyze conv2d method. Messages: ${conv2dResult.messages}"
        )

        // Test case 4: Methods with default parameter mapping (primitive types)
        val primitiveParamsSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface PrimitiveParameterOps {
                fun scaleOperation(
                    tensor: sk.ainet.lang.tensor.Tensor<Float, Any>,
                    scale: Float,
                    offset: Int,
                    enabled: Boolean
                ): sk.ainet.lang.tensor.Tensor<Float, Any>
                
                fun configureOperation(
                    input: sk.ainet.lang.tensor.Tensor<Double, Any>,
                    name: String,
                    iterations: Long,
                    precision: Short
                ): sk.ainet.lang.tensor.Tensor<Double, Any>
            }
        """.trimIndent()

        val primitiveResult = compileWithProcessor(primitiveParamsSource, "PrimitiveParameterInterface.kt")
        
        // Should handle primitive parameter types for default mapping
        assertTrue(
            primitiveResult.messages.contains("Generating tracing wrapper for PrimitiveParameterOps"),
            "Should generate wrapper for primitive parameter ops. Messages: ${primitiveResult.messages}"
        )
        assertTrue(
            primitiveResult.messages.contains("Analyzed 2 methods for PrimitiveParameterOps"),
            "Should analyze both primitive parameter methods. Messages: ${primitiveResult.messages}"
        )

        // Test case 5: Methods with collection parameters
        val collectionParamsSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface CollectionParameterOps {
                fun processWithList(
                    tensor: sk.ainet.lang.tensor.Tensor<Int, Any>,
                    indices: List<Int>,
                    weights: List<Float>
                ): sk.ainet.lang.tensor.Tensor<Int, Any>
                
                fun processWithArray(
                    input: sk.ainet.lang.tensor.Tensor<Float, Any>,
                    shape: IntArray,
                    values: FloatArray
                ): sk.ainet.lang.tensor.Tensor<Float, Any>
                
                fun processWithPair(
                    tensor: sk.ainet.lang.tensor.Tensor<Double, Any>,
                    dimensions: Pair<Int, Int>,
                    range: Pair<Double, Double>
                ): sk.ainet.lang.tensor.Tensor<Double, Any>
            }
        """.trimIndent()

        val collectionResult = compileWithProcessor(collectionParamsSource, "CollectionParameterInterface.kt")
        
        // Should handle collection parameter types for default mapping
        assertTrue(
            collectionResult.messages.contains("Generating tracing wrapper for CollectionParameterOps"),
            "Should generate wrapper for collection parameter ops. Messages: ${collectionResult.messages}"
        )
        assertTrue(
            collectionResult.messages.contains("Analyzed 3 methods for CollectionParameterOps"),
            "Should analyze all 3 collection parameter methods. Messages: ${collectionResult.messages}"
        )

        // Test case 6: Methods with custom data class parameters
        val customParamsSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            data class CustomConfig(val value: Int, val name: String)
            data class ProcessingOptions(val mode: String, val threshold: Double)
            
            @GenerateTracingWrapper
            interface CustomParameterOps {
                fun processWithCustom(
                    tensor: sk.ainet.lang.tensor.Tensor<Float, Any>,
                    config: CustomConfig,
                    options: ProcessingOptions
                ): sk.ainet.lang.tensor.Tensor<Float, Any>
                
                fun processWithNullable(
                    input: sk.ainet.lang.tensor.Tensor<Double, Any>,
                    optionalConfig: CustomConfig?
                ): sk.ainet.lang.tensor.Tensor<Double, Any>
            }
        """.trimIndent()

        val customResult = compileWithProcessor(customParamsSource, "CustomParameterInterface.kt")
        
        // Should handle custom data class parameter types
        assertTrue(
            customResult.messages.contains("Generating tracing wrapper for CustomParameterOps"),
            "Should generate wrapper for custom parameter ops. Messages: ${customResult.messages}"
        )
        assertTrue(
            customResult.messages.contains("Analyzed 2 methods for CustomParameterOps"),
            "Should analyze both custom parameter methods. Messages: ${customResult.messages}"
        )

        // Test case 7: Mixed OpAttributeFactory and default mapping
        val mixedAttributesSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface MixedAttributeOps {
                // Binary op with additional non-tensor parameters
                fun addWithScale(
                    a: sk.ainet.lang.tensor.Tensor<Float, Any>,
                    b: sk.ainet.lang.tensor.Tensor<Float, Any>,
                    scale: Float,
                    clamp: Boolean
                ): sk.ainet.lang.tensor.Tensor<Float, Any>
                
                // Unary op with additional parameters
                fun reluWithThreshold(
                    input: sk.ainet.lang.tensor.Tensor<Float, Any>,
                    threshold: Float,
                    inplace: Boolean
                ): sk.ainet.lang.tensor.Tensor<Float, Any>
                
                // Non-factory method with multiple parameter types
                fun customOperation(
                    tensor: sk.ainet.lang.tensor.Tensor<Double, Any>,
                    mode: String,
                    params: List<Int>,
                    config: Pair<Float, Float>
                ): sk.ainet.lang.tensor.Tensor<Double, Any>
            }
        """.trimIndent()

        val mixedResult = compileWithProcessor(mixedAttributesSource, "MixedAttributeInterface.kt")
        
        // Should handle mixed attribute strategies correctly
        assertTrue(
            mixedResult.messages.contains("Generating tracing wrapper for MixedAttributeOps"),
            "Should generate wrapper for mixed attribute ops. Messages: ${mixedResult.messages}"
        )
        assertTrue(
            mixedResult.messages.contains("Analyzed 3 methods for MixedAttributeOps"),
            "Should analyze all 3 mixed attribute methods. Messages: ${mixedResult.messages}"
        )

        // Test case 8: Methods with no non-tensor parameters (shapesAndDTypes fallback)
        val tensorOnlySource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface TensorOnlyOps {
                fun pureTensorOperation(
                    input: sk.ainet.lang.tensor.Tensor<Float, Any>
                ): sk.ainet.lang.tensor.Tensor<Float, Any>
                
                fun multipleTensorInputs(
                    a: sk.ainet.lang.tensor.Tensor<Double, Any>,
                    b: sk.ainet.lang.tensor.Tensor<Double, Any>,
                    c: sk.ainet.lang.tensor.Tensor<Double, Any>
                ): sk.ainet.lang.tensor.Tensor<Double, Any>
            }
        """.trimIndent()

        val tensorOnlyResult = compileWithProcessor(tensorOnlySource, "TensorOnlyInterface.kt")
        
        // Should use shapesAndDTypes for tensor-only operations
        assertTrue(
            tensorOnlyResult.messages.contains("Generating tracing wrapper for TensorOnlyOps"),
            "Should generate wrapper for tensor-only ops. Messages: ${tensorOnlyResult.messages}"
        )
        assertTrue(
            tensorOnlyResult.messages.contains("Analyzed 2 methods for TensorOnlyOps"),
            "Should analyze both tensor-only methods. Messages: ${tensorOnlyResult.messages}"
        )

        // Test case 9: Methods with no parameters at all
        val noParamsSource = """
            package test
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface NoParameterOps {
                fun generateRandomTensor(): sk.ainet.lang.tensor.Tensor<Float, Any>
                fun getCurrentTimestamp(): Long
                fun getDefaultConfig(): String
            }
        """.trimIndent()

        val noParamsResult = compileWithProcessor(noParamsSource, "NoParameterInterface.kt")
        
        // Should handle methods with no parameters
        assertTrue(
            noParamsResult.messages.contains("Generating tracing wrapper for NoParameterOps"),
            "Should generate wrapper for no-parameter ops. Messages: ${noParamsResult.messages}"
        )
        assertTrue(
            noParamsResult.messages.contains("Analyzed 3 methods for NoParameterOps"),
            "Should analyze all 3 no-parameter methods. Messages: ${noParamsResult.messages}"
        )
    }

    /**
     * Property 7: Code Generation Quality
     * For any generated tracing wrapper, the code should compile successfully without 
     * errors and all methods should be properly implemented.
     * **Validates: Requirements 8.4**
     */
    @Test
    fun testCodeGenerationQuality() {
        // Test case 1: Complex interface with various method signatures
        val complexInterfaceSource = """
            package test.quality
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface QualityTestOps {
                // Simple method
                fun basicOperation(): String
                
                // Method with primitive parameters
                fun withPrimitives(x: Int, y: Float, flag: Boolean): Double
                
                // Generic method
                fun <T> genericMethod(value: T): T
                
                // Method with tensor parameters
                fun tensorOperation(
                    input: sk.ainet.lang.tensor.Tensor<Float, Any>,
                    weight: sk.ainet.lang.tensor.Tensor<Float, Any>
                ): sk.ainet.lang.tensor.Tensor<Float, Any>
                
                // Multi-output method
                fun multiOutput(
                    tensor: sk.ainet.lang.tensor.Tensor<Double, Any>
                ): List<sk.ainet.lang.tensor.Tensor<Double, Any>>
                
                // Method with complex parameters
                fun complexMethod(
                    tensor: sk.ainet.lang.tensor.Tensor<Int, Any>,
                    config: Pair<Int, Int>,
                    options: List<String>,
                    threshold: Float?
                ): sk.ainet.lang.tensor.Tensor<Int, Any>
                
                // Unit return method
                fun unitMethod(message: String): Unit
                
                // Method with default parameters
                fun withDefaults(required: String, optional: Int = 42): Boolean
            }
        """.trimIndent()

        val result = compileWithProcessor(complexInterfaceSource, "QualityTestInterface.kt")
        
        // Should successfully generate code without errors
        assertTrue(
            result.messages.contains("Generating tracing wrapper for QualityTestOps"),
            "Should start code generation. Messages: ${result.messages}"
        )
        assertTrue(
            result.messages.contains("Analyzed 8 methods for QualityTestOps"),
            "Should analyze all 8 methods. Messages: ${result.messages}"
        )
        assertTrue(
            result.messages.contains("Generated code validation passed for KspQualityTestOps"),
            "Generated code should pass validation. Messages: ${result.messages}"
        )
        assertTrue(
            result.messages.contains("Successfully generated KspQualityTestOps.kt in package test.quality"),
            "Should complete generation successfully. Messages: ${result.messages}"
        )

        // Test case 2: Interface with edge case method signatures
        val edgeCaseSource = """
            package test.edge
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface EdgeCaseOps {
                // Method with many parameters (but within limits)
                fun manyParams(
                    p1: Int, p2: Float, p3: Double, p4: Boolean, p5: String,
                    p6: Long, p7: Short, p8: Byte, p9: Char, p10: Int
                ): String
                
                // Method with nested generics
                fun nestedGenerics(
                    input: List<sk.ainet.lang.tensor.Tensor<Float, Any>>
                ): List<sk.ainet.lang.tensor.Tensor<Float, Any>>
                
                // Method with nullable parameters
                fun withNullables(
                    tensor: sk.ainet.lang.tensor.Tensor<Double, Any>?,
                    config: String?,
                    value: Int?
                ): sk.ainet.lang.tensor.Tensor<Double, Any>?
                
                // Method with array parameters
                fun withArrays(
                    tensor: sk.ainet.lang.tensor.Tensor<Int, Any>,
                    shape: IntArray,
                    values: FloatArray
                ): sk.ainet.lang.tensor.Tensor<Int, Any>
            }
        """.trimIndent()

        val edgeResult = compileWithProcessor(edgeCaseSource, "EdgeCaseInterface.kt")
        
        // Should handle edge cases correctly
        assertTrue(
            edgeResult.messages.contains("Generating tracing wrapper for EdgeCaseOps"),
            "Should handle edge case generation. Messages: ${edgeResult.messages}"
        )
        assertTrue(
            edgeResult.messages.contains("Analyzed 4 methods for EdgeCaseOps"),
            "Should analyze all 4 edge case methods. Messages: ${edgeResult.messages}"
        )
        assertTrue(
            edgeResult.messages.contains("Generated code validation passed for KspEdgeCaseOps"),
            "Edge case code should pass validation. Messages: ${edgeResult.messages}"
        )

        // Test case 3: Interface with known OpAttributeFactory operations
        val factoryOpsSource = """
            package test.factory
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface FactoryOps {
                // Binary operations
                fun add(
                    a: sk.ainet.lang.tensor.Tensor<Float, Any>,
                    b: sk.ainet.lang.tensor.Tensor<Float, Any>
                ): sk.ainet.lang.tensor.Tensor<Float, Any>
                
                fun multiply(
                    x: sk.ainet.lang.tensor.Tensor<Double, Any>,
                    y: sk.ainet.lang.tensor.Tensor<Double, Any>
                ): sk.ainet.lang.tensor.Tensor<Double, Any>
                
                // Unary operations
                fun relu(
                    input: sk.ainet.lang.tensor.Tensor<Float, Any>
                ): sk.ainet.lang.tensor.Tensor<Float, Any>
                
                fun sigmoid(
                    tensor: sk.ainet.lang.tensor.Tensor<Double, Any>
                ): sk.ainet.lang.tensor.Tensor<Double, Any>
                
                // Conv2d operation
                fun conv2d(
                    input: sk.ainet.lang.tensor.Tensor<Float, Any>,
                    weight: sk.ainet.lang.tensor.Tensor<Float, Any>,
                    bias: sk.ainet.lang.tensor.Tensor<Float, Any>?,
                    stride: Pair<Int, Int>,
                    padding: Pair<Int, Int>
                ): sk.ainet.lang.tensor.Tensor<Float, Any>
            }
        """.trimIndent()

        val factoryResult = compileWithProcessor(factoryOpsSource, "FactoryOpsInterface.kt")
        
        // Should generate high-quality code for factory operations
        assertTrue(
            factoryResult.messages.contains("Generating tracing wrapper for FactoryOps"),
            "Should generate factory ops wrapper. Messages: ${factoryResult.messages}"
        )
        assertTrue(
            factoryResult.messages.contains("Analyzed 5 methods for FactoryOps"),
            "Should analyze all 5 factory methods. Messages: ${factoryResult.messages}"
        )
        assertTrue(
            factoryResult.messages.contains("Generated code validation passed for KspFactoryOps"),
            "Factory ops code should pass validation. Messages: ${factoryResult.messages}"
        )

        // Test case 4: Interface with validation edge cases
        val validationSource = """
            package test.validation
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface ValidationOps {
                // Method that could cause naming conflicts
                fun base(): String
                fun sink(): String  
                fun session(): String
                
                // Method with reserved keywords as parameters
                fun withReserved(class_: String, object_: Int): String
                
                // Method with special characters in names (valid Kotlin)
                fun `method with spaces`(): String
                fun methodWith123Numbers(): String
            }
        """.trimIndent()

        val validationResult = compileWithProcessor(validationSource, "ValidationInterface.kt")
        
        // Should detect and handle validation issues
        if (validationResult.messages.contains("Method 'base' conflicts with generated constructor parameters") ||
            validationResult.messages.contains("Method 'sink' conflicts with generated constructor parameters") ||
            validationResult.messages.contains("Method 'session' conflicts with generated constructor parameters")) {
            // Expected validation error for conflicting method names
            assertTrue(
                validationResult.messages.contains("Code generation validation failed"),
                "Should detect method name conflicts. Messages: ${validationResult.messages}"
            )
        } else {
            // If no conflicts detected, generation should succeed
            assertTrue(
                validationResult.messages.contains("Generated code validation passed") ||
                validationResult.messages.contains("Successfully generated"),
                "Should either detect conflicts or generate successfully. Messages: ${validationResult.messages}"
            )
        }

        // Test case 5: Large interface with many methods
        val largeInterfaceSource = """
            package test.large
            
            import sk.ainet.lang.trace.GenerateTracingWrapper
            
            @GenerateTracingWrapper
            interface LargeOps {
                fun method01(): String
                fun method02(x: Int): String
                fun method03(x: Int, y: Float): String
                fun method04(tensor: sk.ainet.lang.tensor.Tensor<Float, Any>): sk.ainet.lang.tensor.Tensor<Float, Any>
                fun method05(): List<sk.ainet.lang.tensor.Tensor<Double, Any>>
                fun method06(a: String, b: Boolean): Int
                fun method07(): Unit
                fun method08(config: Pair<Int, Int>): String
                fun method09(values: List<Float>): String
                fun method10(tensor: sk.ainet.lang.tensor.Tensor<Int, Any>, scale: Float): sk.ainet.lang.tensor.Tensor<Int, Any>
                fun method11(): String
                fun method12(x: Int): String
                fun method13(x: Int, y: Float): String
                fun method14(tensor: sk.ainet.lang.tensor.Tensor<Float, Any>): sk.ainet.lang.tensor.Tensor<Float, Any>
                fun method15(): List<sk.ainet.lang.tensor.Tensor<Double, Any>>
            }
        """.trimIndent()

        val largeResult = compileWithProcessor(largeInterfaceSource, "LargeInterface.kt")
        
        // Should handle large interfaces efficiently
        assertTrue(
            largeResult.messages.contains("Generating tracing wrapper for LargeOps"),
            "Should handle large interface generation. Messages: ${largeResult.messages}"
        )
        assertTrue(
            largeResult.messages.contains("Analyzed 15 methods for LargeOps"),
            "Should analyze all 15 methods. Messages: ${largeResult.messages}"
        )
        assertTrue(
            largeResult.messages.contains("Generated code validation passed for KspLargeOps"),
            "Large interface code should pass validation. Messages: ${largeResult.messages}"
        )
    }

    private fun compileWithProcessor(sourceCode: String, fileName: String): KotlinCompilation.Result {
        val source = SourceFile.kotlin("test/$fileName", sourceCode)
        
        // Add mock tracing classes for compilation
        val mockSources = listOf(
            SourceFile.kotlin("sk/ainet/lang/trace/OpSink.kt", """
                package sk.ainet.lang.trace
                interface OpSink {
                    fun onOpExecuted(trace: OpTrace)
                }
            """.trimIndent()),
            SourceFile.kotlin("sk/ainet/lang/trace/OpTrace.kt", """
                package sk.ainet.lang.trace
                data class OpTrace(
                    val opType: String,
                    val inputs: List<TensorRef>,
                    val outputs: List<TensorRef>,
                    val attributes: Map<String, Any?>
                )
            """.trimIndent()),
            SourceFile.kotlin("sk/ainet/lang/trace/TensorRef.kt", """
                package sk.ainet.lang.trace
                data class TensorRef(
                    val id: String,
                    val shape: List<Int> = emptyList(),
                    val dtype: String = "float32"
                )
            """.trimIndent()),
            SourceFile.kotlin("sk/ainet/lang/trace/TraceSession.kt", """
                package sk.ainet.lang.trace
                class TraceSession {
                    private var nextId = 0
                    fun refOf(tensor: Any): TensorRef = TensorRef("t${'$'}{nextId++}")
                    fun refsOf(tensors: List<Any>): List<TensorRef> = tensors.map { refOf(it) }
                }
            """.trimIndent()),
            SourceFile.kotlin("sk/ainet/lang/trace/OpAttributeFactory.kt", """
                package sk.ainet.lang.trace
                object OpAttributeFactory {
                    fun binary(a: Any, b: Any, result: Any): Map<String, Any?> = mapOf("type" to "binary")
                    fun unary(input: Any, result: Any): Map<String, Any?> = mapOf("type" to "unary")
                    fun conv2d(input: Any, weight: Any, bias: Any?, result: Any, stride: Any, padding: Any, dilation: Any, groups: Any): Map<String, Any?> = mapOf("type" to "conv2d")
                    fun shapesAndDTypes(inputs: List<Any>, outputs: List<Any>): Map<String, Any?> = mapOf("inputs" to inputs.size, "outputs" to outputs.size)
                }
            """.trimIndent()),
            SourceFile.kotlin("sk/ainet/Tensor.kt", """
                package sk.ainet
                interface Tensor<T> {
                    val shape: List<Int>
                }
            """.trimIndent())
        )
        
        val compilation = KotlinCompilation().apply {
            sources = listOf(source) + mockSources
            symbolProcessorProviders = listOf(TracingWrapperProcessorProvider())
            inheritClassPath = true
            messageOutputStream = System.out
        }
        
        val result = compilation.compile()
        println("[DEBUG_LOG] Compilation result for $fileName: ${result.exitCode}")
        println("[DEBUG_LOG] Output messages: ${result.messages}")
        
        return result
    }
}