package sk.ainet.apps.grayscale

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32

/**
 * Pipeline for converting loaded images to tensors and executing grayscale conversion.
 * Handles tensor shape validation and type conversion between different model precisions.
 */
public class TensorConversionPipeline {
    
    /**
     * Converts a loaded image to a tensor and applies grayscale conversion.
     * 
     * @param image The loaded image to process
     * @param modelInstance The grayscale model instance to use
     * @return ProcessingResult containing the grayscale tensor and metadata
     */
    public suspend fun processImage(
        image: LoadedImage,
        modelInstance: GrayscaleModelInstance
    ): ProcessingResult {
        return when (modelInstance) {
            is GrayscaleModelInstance.FP32Model -> {
                processImageFP32(image, modelInstance)
            }
            is GrayscaleModelInstance.FP16Model -> {
                processImageFP16(image, modelInstance)
            }
        }
    }
    
    /**
     * Processes an image using an FP32 model (Rgb2GrayScale).
     */
    private suspend fun processImageFP32(
        image: LoadedImage,
        modelInstance: GrayscaleModelInstance.FP32Model
    ): ProcessingResult {
        val startTime = System.currentTimeMillis()
        
        try {
            // Convert image to tensor using ImageLoader
            val imageLoader = ImageLoader()
            val inputTensor = imageLoader.imageToTensor(image, modelInstance.executionContext)
            
            // Validate input tensor shape (should be 1,3,H,W)
            validateInputTensorShape(inputTensor.shape, image)
            
            // Convert FP16 tensor to FP32 if needed
            val fp32InputTensor = convertToFP32(inputTensor, modelInstance.executionContext)
            
            // Create model module
            val module = modelInstance.model.create(modelInstance.executionContext)
            
            // Execute grayscale conversion
            val outputTensor = modelInstance.model.calculate(
                module = module,
                inputValue = fp32InputTensor,
                executionContext = modelInstance.executionContext
            ) { current, total, message ->
                // Progress reporting - could be used for verbose output
                if (message != null) {
                    // Could log progress if verbose mode is enabled
                }
            }
            
            // Validate output tensor shape (should be 1,1,H,W)
            validateOutputTensorShape(outputTensor.shape, image)
            
            val processingTime = System.currentTimeMillis() - startTime
            
            return ProcessingResult.Success(
                inputPath = image.path,
                outputTensor = TensorResult.FP32Tensor(outputTensor),
                processingTimeMs = processingTime,
                originalSize = image.width to image.height,
                modelType = GrayscaleModelType.RGB2GRAYSCALE
            )
            
        } catch (e: Exception) {
            val processingTime = System.currentTimeMillis() - startTime
            return ProcessingResult.Error(
                inputPath = image.path,
                error = e.message ?: "Unknown error during FP32 processing",
                processingTimeMs = processingTime,
                cause = e
            )
        }
    }
    
    /**
     * Processes an image using an FP16 model (Rgb2GrayScaleMatMul).
     */
    private suspend fun processImageFP16(
        image: LoadedImage,
        modelInstance: GrayscaleModelInstance.FP16Model
    ): ProcessingResult {
        val startTime = System.currentTimeMillis()
        
        try {
            // Convert image to tensor using ImageLoader
            val imageLoader = ImageLoader()
            val inputTensor = imageLoader.imageToTensor(image, modelInstance.executionContext)
            
            // Validate input tensor shape (should be 1,3,H,W)
            validateInputTensorShape(inputTensor.shape, image)
            
            // The imageToTensor should already return FP16 tensor, but let's ensure it
            val fp16InputTensor = ensureFP16(inputTensor, modelInstance.executionContext)
            
            // Create model module
            val module = modelInstance.model.create(modelInstance.executionContext)
            
            // Execute grayscale conversion
            val outputTensor = modelInstance.model.calculate(
                module = module,
                inputValue = fp16InputTensor,
                executionContext = modelInstance.executionContext
            ) { current, total, message ->
                // Progress reporting - could be used for verbose output
                if (message != null) {
                    // Could log progress if verbose mode is enabled
                }
            }
            
            // Validate output tensor shape (should be 1,1,H,W)
            validateOutputTensorShape(outputTensor.shape, image)
            
            val processingTime = System.currentTimeMillis() - startTime
            
            return ProcessingResult.Success(
                inputPath = image.path,
                outputTensor = TensorResult.FP16Tensor(outputTensor),
                processingTimeMs = processingTime,
                originalSize = image.width to image.height,
                modelType = GrayscaleModelType.RGB2GRAYSCALE_MATMUL
            )
            
        } catch (e: Exception) {
            val processingTime = System.currentTimeMillis() - startTime
            return ProcessingResult.Error(
                inputPath = image.path,
                error = e.message ?: "Unknown error during FP16 processing",
                processingTimeMs = processingTime,
                cause = e
            )
        }
    }
    
    /**
     * Validates that the input tensor has the expected shape (1,3,H,W).
     */
    private fun validateInputTensorShape(shape: Shape, image: LoadedImage) {
        val dims = shape.dimensions
        
        if (dims.size != 4) {
            throw TensorShapeException(
                "Expected input tensor to have 4 dimensions (1,3,H,W), but got ${dims.size} dimensions: $shape"
            )
        }
        
        if (dims[0] != 1) {
            throw TensorShapeException(
                "Expected batch size of 1, but got ${dims[0]}"
            )
        }
        
        if (dims[1] != 3) {
            throw TensorShapeException(
                "Expected 3 channels (RGB), but got ${dims[1]} channels"
            )
        }
        
        if (dims[2] != image.height || dims[3] != image.width) {
            throw TensorShapeException(
                "Tensor spatial dimensions (${dims[2]}, ${dims[3]}) don't match image dimensions (${image.height}, ${image.width})"
            )
        }
    }
    
    /**
     * Validates that the output tensor has the expected shape (1,1,H,W).
     */
    private fun validateOutputTensorShape(shape: Shape, image: LoadedImage) {
        val dims = shape.dimensions
        
        if (dims.size != 4) {
            throw TensorShapeException(
                "Expected output tensor to have 4 dimensions (1,1,H,W), but got ${dims.size} dimensions: $shape"
            )
        }
        
        if (dims[0] != 1) {
            throw TensorShapeException(
                "Expected batch size of 1, but got ${dims[0]}"
            )
        }
        
        if (dims[1] != 1) {
            throw TensorShapeException(
                "Expected 1 channel (grayscale), but got ${dims[1]} channels"
            )
        }
        
        if (dims[2] != image.height || dims[3] != image.width) {
            throw TensorShapeException(
                "Output tensor spatial dimensions (${dims[2]}, ${dims[3]}) don't match image dimensions (${image.height}, ${image.width})"
            )
        }
    }
    
    /**
     * Converts an FP16 tensor to FP32 for use with FP32 models.
     */
    private fun convertToFP32(
        tensor: Tensor<FP16, Float>,
        context: ExecutionContext
    ): Tensor<FP32, Float> {
        // Extract the float array from the tensor data
        val floatData = (tensor.data as FloatArrayTensorData<FP16>).buffer
        
        // Create a new FP32 tensor with the same shape and data
        return context.fromFloatArray<FP32, Float>(
            shape = tensor.shape,
            dtype = FP32::class,
            data = floatData
        )
    }
    
    /**
     * Ensures the tensor is FP16 type (currently assumes it already is).
     */
    private fun ensureFP16(
        tensor: Tensor<FP16, Float>,
        context: ExecutionContext
    ): Tensor<FP16, Float> {
        // The tensor should already be FP16 from imageToTensor
        return tensor
    }
}

/**
 * Sealed class representing the result of tensor processing.
 */
public sealed class ProcessingResult {
    public abstract val inputPath: String
    public abstract val processingTimeMs: Long
    
    public data class Success(
        override val inputPath: String,
        val outputTensor: TensorResult,
        override val processingTimeMs: Long,
        val originalSize: Pair<Int, Int>,
        val modelType: GrayscaleModelType
    ) : ProcessingResult()
    
    public data class Error(
        override val inputPath: String,
        val error: String,
        override val processingTimeMs: Long,
        val cause: Throwable? = null
    ) : ProcessingResult()
}

/**
 * Sealed class for holding tensors of different precision types.
 */
public sealed class TensorResult {
    public data class FP32Tensor(val tensor: Tensor<FP32, Float>) : TensorResult()
    public data class FP16Tensor(val tensor: Tensor<FP16, Float>) : TensorResult()
}

/**
 * Exception thrown when tensor shapes don't match expectations.
 */
public class TensorShapeException(message: String) : Exception(message)