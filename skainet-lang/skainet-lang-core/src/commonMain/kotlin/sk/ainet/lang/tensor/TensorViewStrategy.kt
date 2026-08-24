@file:Suppress("DEPRECATION") // the pre-#1034 view mechanism: kept working until the next major

package sk.ainet.lang.tensor

import sk.ainet.lang.types.DType

/**
 * Strategy interface for deciding between view creation and tensor copying.
 * 
 * This interface provides the foundation for implementing fallback mechanisms
 * when tensor views are not suitable or efficient for particular slicing operations.
 * The strategy helps balance memory efficiency with performance considerations.
 * 
 * ## Use Cases
 * 
 * - **Complex Non-Contiguous Slices**: When slice patterns result in very 
 *   inefficient memory access patterns
 * - **Memory Pressure**: When system memory is under pressure and views 
 *   might prevent garbage collection of large parent tensors
 * - **Backend Limitations**: When the underlying compute backend doesn't 
 *   support efficient view operations
 * 
 * @param T the data type constraint extending DType
 * @param V the actual value type
 */
public interface TensorViewStrategy<T : DType, V> {
    
    /**
     * Determines whether a view should be created for the given slices.
     * 
     * This method analyzes the slice configuration and current system state
     * to decide if creating a view is preferable to copying data. The decision
     * should consider:
     * 
     * - Slice contiguity and access pattern efficiency
     * - Memory usage implications
     * - Backend capabilities and limitations
     * - Performance characteristics of the operation
     * 
     * @param tensor the parent tensor being sliced
     * @param slices the list of slice descriptors
     * @return true if a view should be created, false if data should be copied
     */
    public fun shouldCreateView(tensor: Tensor<T, V>, slices: List<Slice<T, V>>): Boolean
    
    /**
     * Provides a reason code for the view vs copy decision.
     * 
     * This method returns a human-readable explanation for why the strategy
     * chose to create a view or perform a copy. Useful for debugging and
     * performance analysis.
     * 
     * @param tensor the parent tensor being sliced
     * @param slices the list of slice descriptors
     * @return a string describing the decision rationale
     */
    public fun getDecisionReason(tensor: Tensor<T, V>, slices: List<Slice<T, V>>): String
}

/**
 * Default implementation of TensorViewStrategy that prefers views when possible.
 * 
 * This strategy implements a conservative approach that favors zero-copy views
 * for most operations while falling back to copying only in specific cases
 * where views would be significantly inefficient.
 * 
 * ## Decision Logic
 * 
 * The strategy creates views when:
 * - Slices are contiguous or have reasonable stride patterns
 * - The parent tensor is not excessively large relative to the view
 * - The slice doesn't result in highly scattered memory access
 * 
 * The strategy falls back to copying when:
 * - Slices would result in very inefficient strided access patterns
 * - Memory pressure is detected (not implemented in this basic version)
 * - The view would be much smaller than the parent (preventing GC)
 */
public open class DefaultTensorViewStrategy<T : DType, V> : TensorViewStrategy<T, V> {
    
    /**
     * Threshold for view size relative to parent size.
     * If the view is smaller than this fraction of the parent,
     * copying might be preferred to allow parent GC.
     */
    public var viewSizeThreshold: Double = 0.1
    
    /**
     * Maximum stride factor before preferring copy over view.
     * If any dimension has a stride larger than this factor
     * times the base dimension size, copying is preferred.
     */
    public var maxStrideThreshold: Int = 10
    
    override fun shouldCreateView(tensor: Tensor<T, V>, slices: List<Slice<T, V>>): Boolean {
        // Check if any slice has excessive stride
        val hasExcessiveStride = slices.any { slice ->
            when (slice) {
                is Slice.Step<*, *> -> slice.step > maxStrideThreshold
                else -> false
            }
        }
        
        if (hasExcessiveStride) {
            return false
        }
        
        // Calculate approximate view size relative to parent
        val parentVolume = tensor.shape.volume
        var viewVolume = 1L
        
        slices.forEachIndexed { index, slice ->
            val dimensionSize = tensor.shape[index]
            viewVolume *= when (slice) {
                is Slice.Range<T, V> -> (slice.end - slice.start).toLong()
                is Slice.At<T, V> -> 1L
                is Slice.All<T, V> -> dimensionSize.toLong()
                is Slice.Step<T, V> -> ((slice.end - slice.start + slice.step - 1) / slice.step).toLong()
            }
        }
        
        val viewSizeRatio = viewVolume.toDouble() / parentVolume.toDouble()
        
        // Prefer view if size ratio is reasonable
        return viewSizeRatio >= viewSizeThreshold
    }
    
    override fun getDecisionReason(tensor: Tensor<T, V>, slices: List<Slice<T, V>>): String {
        val hasExcessiveStride = slices.any { slice ->
            when (slice) {
                is Slice.Step<*, *> -> slice.step > maxStrideThreshold
                else -> false
            }
        }
        
        if (hasExcessiveStride) {
            return "Excessive stride detected in slice pattern, copying preferred for efficiency"
        }
        
        val parentVolume = tensor.shape.volume
        var viewVolume = 1L
        
        slices.forEachIndexed { index, slice ->
            val dimensionSize = tensor.shape[index]
            viewVolume *= when (slice) {
                is Slice.Range<T, V> -> (slice.end - slice.start).toLong()
                is Slice.At<T, V> -> 1L
                is Slice.All<T, V> -> dimensionSize.toLong()
                is Slice.Step<T, V> -> ((slice.end - slice.start + slice.step - 1) / slice.step).toLong()
            }
        }
        
        val viewSizeRatio = viewVolume.toDouble() / parentVolume.toDouble()
        val ratioRounded = (viewSizeRatio * 100).toInt() / 100.0
        
        return if (viewSizeRatio >= viewSizeThreshold) {
            "View size ratio ($ratioRounded) is reasonable, zero-copy view preferred"
        } else {
            "View size ratio ($ratioRounded) is small, copying preferred to enable parent GC"
        }
    }
}

/**
 * Memory-pressure-aware tensor view strategy.
 * 
 * This strategy extends the default strategy with basic memory pressure detection.
 * When memory pressure is high, it more aggressively prefers copying over views
 * to allow garbage collection of parent tensors.
 */
public class MemoryAwareTensorViewStrategy<T : DType, V> : DefaultTensorViewStrategy<T, V>() {
    
    /**
     * Threshold for available memory percentage.
     * If available memory falls below this percentage,
     * the strategy becomes more aggressive about copying.
     */
    public var memoryPressureThreshold: Double = 0.2
    
    /**
     * Adjusted view size threshold under memory pressure.
     */
    public var pressureViewSizeThreshold: Double = 0.3
    
    override fun shouldCreateView(tensor: Tensor<T, V>, slices: List<Slice<T, V>>): Boolean {
        // Use memory pressure detection (simplified - in real implementation 
        // this would check actual memory usage)
        val isUnderMemoryPressure = isMemoryPressureDetected()
        
        if (isUnderMemoryPressure) {
            // Use more restrictive threshold under memory pressure
            val originalThreshold = viewSizeThreshold
            viewSizeThreshold = pressureViewSizeThreshold
            val result = super.shouldCreateView(tensor, slices)
            viewSizeThreshold = originalThreshold
            return result
        }
        
        return super.shouldCreateView(tensor, slices)
    }
    
    override fun getDecisionReason(tensor: Tensor<T, V>, slices: List<Slice<T, V>>): String {
        val isUnderMemoryPressure = isMemoryPressureDetected()
        val baseReason = super.getDecisionReason(tensor, slices)
        
        return if (isUnderMemoryPressure) {
            "Memory pressure detected. $baseReason"
        } else {
            baseReason
        }
    }
    
    /**
     * Simplified memory pressure detection.
     * In a real implementation, this would check system memory usage,
     * JVM heap usage, or other relevant metrics.
     */
    private fun isMemoryPressureDetected(): Boolean {
        // Placeholder implementation - always returns false
        // Real implementation would check memory usage
        return false
    }
}