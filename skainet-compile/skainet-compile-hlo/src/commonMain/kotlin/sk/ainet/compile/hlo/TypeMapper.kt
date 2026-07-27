package sk.ainet.compile.hlo

import sk.ainet.lang.tensor.Dim
import sk.ainet.lang.tensor.ops.TensorSpec

/**
 * Handles type system mapping between SKaiNET and MLIR types.
 * 
 * This class provides conversion utilities for mapping SKaiNET data types
 * and tensor specifications to their corresponding MLIR representations.
 */
public class TypeMapper {
    
    /**
     * Map SKaiNET data type string to MLIR element type
     */
    public fun mapDType(dtype: String): String = when (dtype.uppercase()) {
        "FP32", "F32", "FLOAT32" -> "f32"
        "FP64", "F64", "FLOAT64" -> "f64"
        "I32", "INT32" -> "i32"
        "I64", "INT64" -> "i64"
        "I8", "INT8" -> "i8"
        "I16", "INT16" -> "i16"
        "UI8", "UINT8" -> "ui8"
        "UI16", "UINT16" -> "ui16"
        "UI32", "UINT32" -> "ui32"
        "UI64", "UINT64" -> "ui64"
        "BF16", "BFLOAT16" -> "bf16"
        "FP16", "F16", "FLOAT16" -> "f16"
        "BOOL", "BOOLEAN" -> "i1"
        else -> {
            // Default fallback with warning
            "f32" // Could emit warning here
        }
    }
    
    /**
     * Bit-pattern literal for -inf in the given MLIR float element type. The
     * width MUST match the type — a 32-bit `0xFF800000` in a `bf16` constant is
     * "out of range" to iree-compile. Used as the identity for `stablehlo.maximum`
     * (softmax / attention max-reduce).
     */
    public fun negInfBits(mlirElementType: String): String = when (mlirElementType) {
        "f64" -> "0xFFF0000000000000"
        "f16" -> "0xFC00"
        "bf16" -> "0xFF80"
        else -> "0xFF800000" // f32 and fallback
    }

    /**
     * Map a TensorSpec to MLIR tensor type string
     */
    public fun mapTensorType(spec: TensorSpec): String {
        val elementType = mapDType(spec.dtype)
        val shapeStr = formatShape(spec.shape)
        // Rank-0 (scalar) is `tensor<elem>`, not `tensor<xelem>` — no leading `x`.
        return if (shapeStr.isEmpty()) "tensor<$elementType>" else "tensor<${shapeStr}x${elementType}>"
    }
    
    /**
     * Map function signature with inputs and outputs
     */
    public fun mapFunctionSignature(inputs: List<TensorSpec>, outputs: List<TensorSpec>): String {
        val inputTypes = inputs.mapIndexed { idx, spec ->
            "%arg$idx: ${mapTensorType(spec)}"
        }.joinToString(", ")
        
        val outputTypes = if (outputs.isEmpty()) {
            "()"
        } else {
            outputs.joinToString(", ") { mapTensorType(it) }
            "(" + outputs.joinToString(", ") { mapTensorType(it) } + ")"
        }
        
        return "($inputTypes) -> $outputTypes"
    }
    
    /**
     * Infer broadcast type for two tensor specs
     */
    public fun inferBroadcastType(left: TensorSpec, right: TensorSpec): TensorSpec {
        // Simple broadcast inference - take the larger shape
        val leftShape = left.shape ?: emptyList()
        val rightShape = right.shape ?: emptyList()
        
        val resultShape = if (leftShape.size >= rightShape.size) leftShape else rightShape
        
        // Use the "higher precision" dtype (simple heuristic)
        val resultDtype = when {
            left.dtype.contains("64") || right.dtype.contains("64") -> 
                if (left.dtype.contains("64")) left.dtype else right.dtype
            left.dtype.contains("32") || right.dtype.contains("32") -> 
                if (left.dtype.contains("32")) left.dtype else right.dtype
            else -> left.dtype
        }
        
        return TensorSpec(
            name = "${left.name}_${right.name}_broadcast",
            shape = resultShape,
            dtype = resultDtype
        )
    }
    
    /**
     * Check if two types are compatible for operations
     */
    public fun areTypesCompatible(left: TensorSpec, right: TensorSpec): Boolean {
        // For now, just check if dtypes are the same
        // More sophisticated compatibility checking can be added later
        return mapDType(left.dtype) == mapDType(right.dtype)
    }
    
    /**
     * Get the MLIR type for a scalar constant of the given dtype
     */
    public fun getScalarType(dtype: String): String {
        return mapDType(dtype)
    }
    
    /**
     * Format tensor shape for MLIR. A negative extent renders as `?` (a dynamic dimension — see
     * [DYNAMIC_DIM]); a `null` shape is a fully-dynamic tensor (`?`).
     */
    public fun formatShape(shape: List<Int>?): String {
        return when {
            shape == null -> "?"
            shape.isEmpty() -> ""
            else -> shape.joinToString("x") { Dim.render(it) }
        }
    }

    /**
     * Create a tensor type string with explicit shape. Renders negative extents as `?` (dynamic).
     */
    public fun createTensorType(shape: List<Int>, dtype: String): String {
        val elementType = mapDType(dtype)
        if (shape.isEmpty()) return "tensor<$elementType>" // rank-0 scalar
        return "tensor<${formatShape(shape)}x${elementType}>"
    }

    public companion object {
        /**
         * Sentinel extent meaning "dynamic dimension" (`?`) in a [TensorSpec] shape. Threaded from the trace
         * (e.g. a KV-cache seq dim) so the emitter renders `?` and picks dynamic-shape-safe op forms, instead
         * of the legacy post-emit text substitution. Aliases the canonical [Dim.DYNAMIC] (a reserved sentinel
         * distinct from reshape's `-1` = infer), so tracer and emitter agree on one value.
         */
        public const val DYNAMIC_DIM: Int = Dim.DYNAMIC
    }
    
    /**
     * Create a dynamic tensor type (with unknown dimensions)
     */
    public fun createDynamicTensorType(rank: Int, dtype: String): String {
        val elementType = mapDType(dtype)
        val shapeStr = (1..rank).joinToString("x") { "?" }
        return "tensor<${shapeStr}x${elementType}>"
    }
}