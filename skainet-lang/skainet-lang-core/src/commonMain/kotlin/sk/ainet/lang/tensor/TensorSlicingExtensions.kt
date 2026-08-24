@file:Suppress("DEPRECATION") // the pre-#1034 view mechanism: kept working until the next major

package sk.ainet.lang.tensor

import sk.ainet.lang.types.DType

/**
 * Extension methods for tensor slicing operations.
 *
 * This file provides the main API for tensor slicing, integrating the DSL builder
 * with the tensor interface to enable intuitive slicing operations.
 */

/**
 * Creates a zero-copy view of this tensor using the provided slicing DSL.
 *
 * This method enables fluent syntax for tensor slicing that creates efficient
 * views without copying data. The resulting TensorView shares memory with
 * the parent tensor and applies coordinate transformations for element access.
 *
 * ## Usage Example
 * ```kotlin
 * val view = tensor.sliceView {
 *     segment { range(0, 10) }  // First dimension: indices 0-9
 *     segment { at(5) }         // Second dimension: index 5 (reduces rank)
 *     segment { all() }         // Third dimension: all indices
 *     segment { step(0, 20, 2) } // Fourth dimension: every 2nd index
 * }
 * ```
 *
 * ## Performance Characteristics
 * - **Zero-copy**: No data duplication, constant memory overhead
 * - **Lazy**: Coordinate mapping applied during element access
 * - **Efficient**: Optimized for NCHW layout and contiguous patterns
 *
 * @param builder the DSL block that configures the slicing operation
 * @return a TensorView providing sliced access to this tensor's data
 * @throws IllegalStateException if slice configuration is invalid
 */
public fun <T : DType, V> Tensor<T, V>.sliceView(
    builder: TensorSliceBuilder<T, V>.() -> Unit
): TensorView<T, V> {
    val sliceBuilder = TensorSliceBuilder<T, V>()
    sliceBuilder.builder()
    
    // Validate the slice configuration against tensor shape
    sliceBuilder.validate(this.shape)
    
    val slices = sliceBuilder.build()
    return SlicedTensorView(this, slices)
}

/**
 * Creates a new tensor by copying data from the sliced region of this tensor.
 *
 * This method provides backward compatibility with existing tensor operations
 * that expect independent tensor instances. Unlike sliceView(), this creates
 * a completely separate tensor with copied data.
 *
 * ## Usage Example
 * ```kotlin
 * val copy = tensor.sliceCopy {
 *     segment { range(0, 10) }
 *     segment { all() }
 * }
 * // `copy` is an independent tensor with copied data
 * ```
 *
 * ## Performance Characteristics
 * - **Data copy**: Duplicates selected data into new tensor
 * - **Independent**: No shared memory with parent tensor
 * - **Compatible**: Works with all existing tensor operations
 *
 * @param builder the DSL block that configures the slicing operation
 * @return a new Tensor containing copied data from the sliced region
 * @throws IllegalStateException if slice configuration is invalid
 */
public fun <T : DType, V> Tensor<T, V>.sliceCopy(
    builder: TensorSliceBuilder<T, V>.() -> Unit
): Tensor<T, V> {
    // First create a view to define the slice
    val view = sliceView(builder)
    
    // Then materialize it into a new tensor
    // Note: This requires a materialization strategy, which will be implemented
    // as part of the advanced features. For now, we throw an exception to indicate
    // this functionality is not yet implemented.
    throw NotImplementedError(
        "sliceCopy() requires materialization strategy implementation. " +
        "Use sliceView() for zero-copy slicing, or implement materialization in advanced features."
    )
}

/**
 * Creates a tensor view using a direct list of slice descriptors.
 *
 * This method provides a lower-level API for slicing when the slice descriptors
 * are already available, bypassing the DSL builder. Useful for programmatic
 * slice generation or integration with other systems.
 *
 * ## Usage Example
 * ```kotlin
 * val slices = listOf(
 *     Slice.Range<MyDType, Float>(0, 10),
 *     Slice.At<MyDType, Float>(5),
 *     Slice.All<MyDType, Float>()
 * )
 * val view = tensor.slice(slices)
 * ```
 *
 * @param slices the list of slice descriptors, one for each tensor dimension
 * @return a TensorView providing sliced access to this tensor's data
 * @throws IllegalArgumentException if slice list doesn't match tensor rank
 * @throws IllegalStateException if any slice is invalid for its dimension
 */
public fun <T : DType, V> Tensor<T, V>.slice(slices: List<Slice<T, V>>): TensorView<T, V> {
    // Validate slice count matches tensor rank
    if (slices.size != this.rank) {
        throw IllegalArgumentException(
            "Slice count must match tensor rank: expected ${this.rank}, got ${slices.size}"
        )
    }
    
    // Validate each slice against its corresponding dimension
    slices.forEachIndexed { index, slice ->
        val dimensionSize = this.shape[index]
        if (!slice.isValid(dimensionSize)) {
            throw IllegalStateException(
                "Invalid slice at dimension $index: $slice (dimension size: $dimensionSize)"
            )
        }
    }
    
    return SlicedTensorView(this, slices)
}

/**
 * Convenience method for creating simple range slices across all dimensions.
 *
 * This method provides a quick way to slice all dimensions with ranges,
 * which is a common operation in neural network processing.
 *
 * ## Usage Example
 * ```kotlin
 * // Slice a 4D NCHW tensor: batch[0:2], channels[1:5], height[10:20], width[5:15]
 * val view = tensor.sliceRanges(
 *     0 to 2,   // batch dimension
 *     1 to 5,   // channel dimension  
 *     10 to 20, // height dimension
 *     5 to 15   // width dimension
 * )
 * ```
 *
 * @param ranges pairs of (start, end) for each dimension
 * @return a TensorView with range slices applied to all dimensions
 * @throws IllegalArgumentException if range count doesn't match tensor rank
 */
public fun <T : DType, V> Tensor<T, V>.sliceRanges(vararg ranges: Pair<Int, Int>): TensorView<T, V> {
    if (ranges.size != this.rank) {
        throw IllegalArgumentException(
            "Range count must match tensor rank: expected ${this.rank}, got ${ranges.size}"
        )
    }
    
    val slices = ranges.map { (start, end) -> 
        Slice.Range<T, V>(start, end) 
    }
    
    return slice(slices)
}

/**
 * Convenience method for extracting specific indices across dimensions.
 *
 * This method enables quick extraction of specific slices by providing
 * individual indices for each dimension, which is useful for accessing
 * specific samples, channels, or spatial locations.
 *
 * ## Usage Example
 * ```kotlin
 * // Extract specific batch=0, channel=3 from NCHW tensor (reduces to HW)
 * val view = tensor.sliceAt(0, 3)  // Results in 2D tensor
 * ```
 *
 * @param indices the specific indices to select for each dimension
 * @return a TensorView with reduced dimensionality
 * @throws IllegalArgumentException if index count doesn't match tensor rank
 */
public fun <T : DType, V> Tensor<T, V>.sliceAt(vararg indices: Int): TensorView<T, V> {
    if (indices.size != this.rank) {
        throw IllegalArgumentException(
            "Index count must match tensor rank: expected ${this.rank}, got ${indices.size}"
        )
    }
    
    val slices = indices.map { index -> 
        Slice.At<T, V>(index)
    }
    
    return slice(slices)
}

// === OPTIMIZATION UTILITIES ===

/**
 * Determines if a list of slices results in a contiguous memory access pattern.
 *
 * This function analyzes slice patterns to optimize memory access and enable
 * vectorized operations. It's particularly optimized for NCHW tensor layout
 * commonly used in deep learning.
 *
 * ## Contiguous Patterns (in order of efficiency):
 * 1. **Batch slicing**: `tensor[0:k, :, :, :]` - Most efficient
 * 2. **Channel slicing**: `tensor[:, 0:k, :, :]` - Highly efficient  
 * 3. **Spatial region slicing**: `tensor[:, :, h1:h2, w1:w2]` - Good efficiency
 * 4. **Width-only slicing**: `tensor[:, :, :, w1:w2]` - Moderate efficiency
 *
 * @param slices the list of slice descriptors to analyze
 * @return true if the slicing pattern results in contiguous memory access
 */
public fun <T : DType, V> isContiguousSlice(slices: List<Slice<T, V>>): Boolean {
    if (slices.isEmpty()) return true
    
    // Find the first non-trivial slice (not All and not single At)
    var firstSliceIndex = -1
    for (i in slices.indices) {
        val slice = slices[i]
        when {
            slice is Slice.All -> continue
            slice is Slice.At -> continue  // At slices don't break contiguity
            slice is Slice.Range && slice.start == 0 && slice.end == 1 -> continue  // Single element range
            slice is Slice.Step && slice.step == 1 -> {
                firstSliceIndex = i
                break
            }
            slice is Slice.Range -> {
                firstSliceIndex = i
                break
            }
            slice.hasNonTrivialStride() -> return false  // Non-unit strides break contiguity
        }
    }
    
    // If no slicing found, it's contiguous
    if (firstSliceIndex == -1) return true
    
    // All slices after the first non-trivial slice must be All or At to maintain contiguity
    for (i in (firstSliceIndex + 1) until slices.size) {
        val slice = slices[i]
        when (slice) {
            is Slice.All -> continue
            is Slice.At -> continue
            else -> return false  // Any other slice type breaks contiguity
        }
    }
    
    return true
}

/**
 * Detects NCHW-optimized slicing patterns for efficient memory access.
 *
 * This function identifies common deep learning slicing patterns that can be
 * highly optimized in NCHW (batch, channel, height, width) tensor layout.
 *
 * @param slices the list of slice descriptors (must be length 4 for NCHW)
 * @return the detected pattern type or null if no optimization applies
 */
public enum class NCHWSlicePattern {
    BATCH_SLICE,    // tensor[0:k, :, :, :] - Most efficient
    CHANNEL_SLICE,  // tensor[:, 0:k, :, :] - Highly efficient
    SPATIAL_REGION, // tensor[:, :, h1:h2, w1:w2] - Good efficiency
    WIDTH_SLICE,    // tensor[:, :, :, w1:w2] - Moderate efficiency
    OTHER           // Non-optimized pattern
}

/**
 * Detects the NCHW slicing pattern for optimization purposes.
 *
 * @param slices the list of slice descriptors for a 4D NCHW tensor
 * @return the detected pattern type
 */
public fun <T : DType, V> detectNCHWSlicePattern(slices: List<Slice<T, V>>): NCHWSlicePattern {
    if (slices.size != 4) return NCHWSlicePattern.OTHER
    
    val (batchSlice, channelSlice, heightSlice, widthSlice) = slices
    
    return when {
        // Batch slicing: [Range/Step, All, All, All]
        (batchSlice is Slice.Range || batchSlice is Slice.Step) &&
        channelSlice is Slice.All &&
        heightSlice is Slice.All &&
        widthSlice is Slice.All -> NCHWSlicePattern.BATCH_SLICE
        
        // Channel slicing: [All/At, Range/Step, All, All]
        (batchSlice is Slice.All || batchSlice is Slice.At) &&
        (channelSlice is Slice.Range || channelSlice is Slice.Step) &&
        heightSlice is Slice.All &&
        widthSlice is Slice.All -> NCHWSlicePattern.CHANNEL_SLICE
        
        // Spatial region: [All/At, All/At, Range/Step, Range/Step]
        (batchSlice is Slice.All || batchSlice is Slice.At) &&
        (channelSlice is Slice.All || channelSlice is Slice.At) &&
        (heightSlice is Slice.Range || heightSlice is Slice.Step) &&
        (widthSlice is Slice.Range || widthSlice is Slice.Step) -> NCHWSlicePattern.SPATIAL_REGION
        
        // Width slicing: [All/At, All/At, All/At, Range/Step]
        (batchSlice is Slice.All || batchSlice is Slice.At) &&
        (channelSlice is Slice.All || channelSlice is Slice.At) &&
        (heightSlice is Slice.All || heightSlice is Slice.At) &&
        (widthSlice is Slice.Range || widthSlice is Slice.Step) -> NCHWSlicePattern.WIDTH_SLICE
        
        else -> NCHWSlicePattern.OTHER
    }
}

// === VIEW COMPOSITION UTILITIES ===

/**
 * Creates a view of an existing tensor view, enabling view chaining.
 *
 * This extension allows creating views of views (view composition) while optimizing
 * the composition by flattening slice operations to prevent deep nesting and
 * maintain performance.
 *
 * ## Usage Example
 * ```kotlin
 * val baseView = tensor.sliceView { 
 *     segment { range(0, 20) }    // First 20 batches
 *     segment { all() }           // All channels
 *     segment { all() }           // All height
 *     segment { all() }           // All width
 * }
 * 
 * val refinedView = baseView.sliceView {
 *     segment { range(5, 15) }    // Batches 5-14 from the base view (actual: 5-14 from original)
 *     segment { range(0, 64) }    // First 64 channels
 *     segment { all() }           // All height  
 *     segment { all() }           // All width
 * }
 * ```
 *
 * ## Optimization Features
 * - **Slice flattening**: Combines multiple slice operations into a single optimized slice
 * - **Bounds validation**: Ensures composed slices remain within valid ranges
 * - **Contiguity preservation**: Maintains memory access optimizations where possible
 * - **Chain optimization**: Prevents excessive view nesting
 *
 * @param builder the DSL block that configures the additional slicing operation
 * @return a TensorView that represents the composed slice operation
 * @throws IllegalStateException if slice composition results in invalid bounds
 * @throws IllegalArgumentException if slice dimensions don't match view rank
 */
public fun <T : DType, V> TensorView<T, V>.sliceView(
    builder: TensorSliceBuilder<T, V>.() -> Unit
): TensorView<T, V> {
    val sliceBuilder = TensorSliceBuilder<T, V>()
    sliceBuilder.builder()
    
    // Validate the slice configuration against view shape (not parent shape)
    sliceBuilder.validate(this.shape)
    
    val newSlices = sliceBuilder.build()
    
    // Optimize composition by flattening slice operations
    val composedSlices = composeSlices(this, newSlices)
    
    // Validate composition doesn't exceed original tensor bounds
    validateComposedSlices(this.parentTensor, composedSlices)
    
    // Create view directly from original parent tensor with composed slices
    return SlicedTensorView(this.parentTensor, composedSlices)
}

/**
 * Composes (flattens) two levels of slicing into a single optimized slice operation.
 *
 * This function takes an existing view's slices and a new set of slices to be applied
 * to that view, then computes the equivalent single slice that would produce the
 * same result when applied directly to the original parent tensor.
 *
 * ## Composition Rules
 * - **Range + Range**: Combine with offset and size adjustment
 * - **Range + At**: Convert to At with offset calculation  
 * - **At + (any)**: Invalid - cannot slice an eliminated dimension
 * - **All + X**: Preserve X unchanged
 * - **Step + Range**: Apply step to range bounds
 * - **Step + Step**: Multiply step values and adjust bounds
 *
 * @param view the existing tensor view with its slice configuration
 * @param newSlices the additional slices to be applied to the view
 * @return the flattened list of slices equivalent to the composition
 * @throws IllegalArgumentException if composition is invalid (e.g., slicing eliminated dimensions)
 */
internal fun <T : DType, V> composeSlices(
    view: TensorView<T, V>,
    newSlices: List<Slice<T, V>>
): List<Slice<T, V>> {
    // Get the current slices from the view
    val currentSlices = when (view) {
        is SlicedTensorView -> view.sliceOperations
        else -> throw IllegalArgumentException("View composition only supported for SlicedTensorView")
    }
    
    if (newSlices.size != view.rank) {
        throw IllegalArgumentException(
            "New slice count must match view rank: expected ${view.rank}, got ${newSlices.size}"
        )
    }
    
    val composedSlices = mutableListOf<Slice<T, V>>()
    var viewDimIndex = 0
    
    // Iterate through original parent dimensions
    for (parentDim in currentSlices.indices) {
        val currentSlice = currentSlices[parentDim]
        
        if (currentSlice is Slice.At) {
            // At slices eliminate dimensions, so they pass through unchanged
            composedSlices.add(currentSlice)
        } else {
            // This dimension exists in the view, so apply the new slice
            if (viewDimIndex >= newSlices.size) {
                throw IllegalArgumentException("Insufficient new slices for view dimensions")
            }
            
            val newSlice = newSlices[viewDimIndex]
            val composed = composeSlicePair(currentSlice, newSlice, view.shape[viewDimIndex])
            composedSlices.add(composed)
            viewDimIndex++
        }
    }
    
    return composedSlices
}

/**
 * Composes two individual slices into a single equivalent slice.
 *
 * @param current the existing slice from the view
 * @param new the new slice to be applied
 * @param viewDimensionSize the size of the dimension in the view
 * @return the composed slice
 */
private fun <T : DType, V> composeSlicePair(
    current: Slice<T, V>,
    new: Slice<T, V>,
    viewDimensionSize: Int
): Slice<T, V> {
    return when (current) {
        is Slice.All -> {
            // All + X = X (unchanged)
            new
        }
        
        is Slice.Range -> when (new) {
            is Slice.All -> current
            is Slice.At -> {
                val actualIndex = current.start + new.index
                Slice.At<T, V>(actualIndex)
            }
            is Slice.Range -> {
                val actualStart = current.start + new.start
                val actualEnd = current.start + new.end
                Slice.Range<T, V>(actualStart, actualEnd)
            }
            is Slice.Step -> {
                val actualStart = current.start + new.start
                val actualEnd = current.start + new.end
                val actualStep = new.step
                Slice.Step<T, V>(actualStart, actualEnd, actualStep)
            }
        }
        
        is Slice.Step -> when (new) {
            is Slice.All -> current
            is Slice.At -> {
                val actualIndex = current.start + (new.index * current.step)
                Slice.At<T, V>(actualIndex)
            }
            is Slice.Range -> {
                val actualStart = current.start + (new.start * current.step)
                val actualEnd = current.start + (new.end * current.step)
                val actualStep = current.step
                Slice.Step<T, V>(actualStart, actualEnd, actualStep)
            }
            is Slice.Step -> {
                val actualStart = current.start + (new.start * current.step)
                val actualEnd = current.start + (new.end * current.step)
                val actualStep = current.step * new.step
                Slice.Step<T, V>(actualStart, actualEnd, actualStep)
            }
        }
        
        is Slice.At -> {
            throw IllegalArgumentException(
                "Cannot apply additional slicing to eliminated dimension (At slice)"
            )
        }
    }
}

/**
 * Validates that composed slices don't exceed the bounds of the original parent tensor.
 *
 * @param parentTensor the original tensor before any slicing
 * @param composedSlices the flattened slices to validate
 * @throws IllegalStateException if any composed slice exceeds parent tensor bounds
 */
private fun <T : DType, V> validateComposedSlices(
    parentTensor: Tensor<T, V>,
    composedSlices: List<Slice<T, V>>
) {
    if (composedSlices.size != parentTensor.rank) {
        throw IllegalStateException(
            "Composed slice count must match parent tensor rank: expected ${parentTensor.rank}, got ${composedSlices.size}"
        )
    }
    
    composedSlices.forEachIndexed { index, slice ->
        val dimensionSize = parentTensor.shape[index]
        if (!slice.isValid(dimensionSize)) {
            throw IllegalStateException(
                "Composed slice at dimension $index exceeds parent tensor bounds: $slice (parent dimension size: $dimensionSize)"
            )
        }
    }
}