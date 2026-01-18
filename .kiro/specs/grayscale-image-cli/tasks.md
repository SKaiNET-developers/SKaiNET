# Implementation Plan: Grayscale Image CLI

## Overview

This implementation plan creates a command-line interface application for converting color images to grayscale using SKaiNET's existing neural network capabilities. The application leverages existing SKaiNET components including the Rgb2GrayScale model, image I/O functionality, and HLO compilation infrastructure to create a high-performance image processing tool.

## Tasks

- [x] 1. Set up CLI application module structure
  - Create new module `skainet-apps/skainet-grayscale-cli`
  - Configure build.gradle.kts with application plugin and dependencies
  - Set up main class and basic project structure
  - Add dependencies: skainet-lang-models, skainet-io-image, skainet-compile-hlo, kotlinx-cli
  - _Requirements: 8.1, 8.2, 8.3_

- [x] 2. Implement CLI argument parsing and configuration
  - [x] 2.1 Create CliConfiguration data class and GrayscaleModelType enum
    - Define configuration structure for input/output paths, model selection, batch mode
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 2.2 Implement argument parser using kotlinx-cli
    - Add --input, --output, --model, --help, --batch flags
    - Implement default output path generation with "_gray" suffix
    - Add validation for required arguments and file paths
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [x] 2.3 Write unit tests for argument parsing
    - Test various command-line argument combinations
    - Test default output path generation
    - Test validation error cases
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 3. Implement image loading and validation
  - [x] 3.1 Create ImageLoader class using skainet-io-image
    - Implement loadImage() using PlatformBitmapImage and platformImageToArgb()
    - Add support for JPEG and PNG formats
    - Implement image format validation and error handling
    - _Requirements: 1.1, 4.1, 4.2, 4.5, 8.2_

  - [x] 3.2 Add batch image loading from directories
    - Implement loadImagesFromDirectory() with recursive traversal
    - Filter supported image formats and skip unsupported files
    - Preserve directory structure for batch processing
    - _Requirements: 7.1, 7.3_

  - [ ]* 3.3 Write unit tests for image loading
    - Test single image loading with various formats
    - Test directory batch loading
    - Test error handling for missing files and unsupported formats
    - _Requirements: 4.1, 4.2, 4.5, 7.1_

- [x] 4. Integrate grayscale conversion models
  - [x] 4.1 Create ModelFactory for grayscale model instantiation
    - Support both Rgb2GrayScale and Rgb2GrayScaleMatMul models
    - Implement model selection based on CLI configuration
    - Create execution context using DirectCpuExecutionContext
    - _Requirements: 1.2, 2.4, 8.1, 8.4_

  - [x] 4.2 Implement tensor conversion pipeline
    - Convert loaded images to tensors using imageToTensor()
    - Execute grayscale conversion using selected model
    - Handle tensor shape validation (1,3,H,W) → (1,1,H,W)
    - _Requirements: 1.1, 1.2, 4.3_

  - [ ] 4.3 Write property test for grayscale conversion correctness
    - **Property 2: Grayscale Conversion Correctness**
    - **Validates: Requirements 1.2**

- [x] 5. Add StableHLO compilation integration
  - [x] 5.1 Implement HLO compilation using skainet-compile-hlo
    - Use toStableHlo() to convert grayscale model to StableHLO MLIR
    - Add compilation validation and error handling
    - Implement fallback to CPU execution on compilation failure
    - _Requirements: 3.1, 3.5, 8.3_

  - [x] 5.2 Create execution context management
    - Implement GPU/CPU execution context selection
    - Add GPU capability detection (placeholder for future IREE integration)
    - Implement automatic fallback from GPU to CPU
    - _Requirements: 3.3, 3.5, 6.4_

  - [ ]* 5.3 Write property test for HLO compilation
    - **Property 8: StableHLO to IREE Compilation**
    - **Validates: Requirements 3.1**

- [x] 6. Implement image saving functionality
  - [x] 6.1 Create ImageSaver class using skainet-io-image
    - Convert grayscale tensors back to images using argbToPlatformImage()
    - Preserve original image format when output format not specified
    - Generate output paths with proper extensions
    - _Requirements: 1.3, 4.4_

  - [x] 6.2 Add batch saving with directory structure preservation
    - Implement batch saving that maintains relative directory structure
    - Handle output directory creation and permission validation
    - Add progress reporting for batch operations
    - _Requirements: 7.2, 7.3_

  - [ ]* 6.3 Write property test for image saving functionality
    - **Property 3: Image Saving Functionality**
    - **Validates: Requirements 1.3**

- [x] 7. Checkpoint - Basic functionality complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement comprehensive error handling
  - [x] 8.1 Create GrayscaleCliError sealed class hierarchy
    - Define ImageLoadError, CompilationError, ExecutionError, SaveError
    - Implement ErrorHandler with helpful error messages and suggestions
    - Add graceful error handling throughout the application
    - _Requirements: 5.1, 5.2, 5.4, 5.5_

  - [x] 8.2 Add dependency and platform validation
    - Implement CUDA driver detection (placeholder for future IREE)
    - Add missing dependency reporting with helpful guidance
    - Ensure CPU-only execution works on all platforms
    - _Requirements: 6.3, 6.4, 6.5_

  - [ ]* 8.3 Write property tests for error handling
    - **Property 14: Missing File Error Handling**
    - **Property 15: Permission Error Handling**
    - **Property 16: Image Loading Error Details**
    - **Validates: Requirements 5.1, 5.2, 5.4**

- [ ] 9. Implement batch processing pipeline
  - [x] 9.1 Create ImageProcessingPipeline class
    - Orchestrate the complete processing workflow
    - Implement single image processing with timing and metadata
    - Add progress reporting and status tracking
    - _Requirements: 1.5, 7.2_

  - [x] 9.2 Add batch processing with error resilience
    - Process multiple images with individual error handling
    - Continue processing remaining images when some fail
    - Generate comprehensive batch processing reports
    - _Requirements: 7.1, 7.4, 7.5_

  - [ ]* 9.3 Write property tests for batch processing
    - **Property 20: Directory Batch Processing**
    - **Property 21: Batch Progress Reporting**
    - **Property 23: Batch Error Resilience**
    - **Validates: Requirements 7.1, 7.2, 7.4**

- [x] 10. Implement main application orchestration
  - [x] 10.1 Create GrayscaleImageCli main class
    - Implement main() function with argument parsing
    - Orchestrate the complete processing pipeline
    - Add processing time reporting and success status
    - _Requirements: 1.5, 2.5_

  - [x] 10.2 Add logging and output formatting
    - Implement consistent logging patterns with other SKaiNET applications
    - Add verbose mode for detailed processing information
    - Format output reports for both single and batch processing
    - _Requirements: 8.5_

  - [ ]* 10.3 Write integration tests for complete pipeline
    - Test end-to-end image processing workflow
    - Test batch processing with realistic directory structures
    - Test error scenarios and fallback mechanisms
    - _Requirements: 1.1, 1.2, 1.3, 7.1, 7.4_

- [x] 11. Add property-based testing suite
  - [x]* 11.1 Write property test for image loading and tensor conversion
    - **Property 1: Image Loading and Tensor Conversion**
    - **Validates: Requirements 1.1**

  - [x]* 11.2 Write property tests for CLI argument parsing
    - **Property 4: Input Path Argument Parsing**
    - **Property 5: Output Path Argument Parsing**
    - **Property 6: Default Output Path Generation**
    - **Property 7: Model Selection Functionality**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

  - [x]* 11.3 Write property tests for format handling
    - **Property 11: Multi-Resolution Image Handling**
    - **Property 12: Format Preservation**
    - **Property 13: Unsupported Format Error Handling**
    - **Validates: Requirements 4.3, 4.4, 4.5**

  - [x]* 11.4 Write property tests for SKaiNET integration
    - **Property 24: SKaiNET Model Integration**
    - **Property 25: Image I/O Module Integration**
    - **Property 26: HLO Compilation Module Integration**
    - **Property 27: Execution Context Consistency**
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4**

- [x] 12. Final checkpoint and documentation
  - [x] 12.1 Create comprehensive README and usage documentation
    - Document installation and usage instructions
    - Add examples for single image and batch processing
    - Document model selection and performance considerations
    - _Requirements: 2.5, 6.3, 6.5_

  - [x] 12.2 Add performance benchmarking and optimization
    - Implement processing time measurement and reporting
    - Add memory usage monitoring where possible
    - Document performance characteristics on different platforms
    - _Requirements: 1.5, 3.4_

  - [x] 12.3 Final testing and validation
    - Ensure all tests pass, ask the user if questions arise.
    - Validate against all requirements
    - Test on multiple platforms (JVM, different OS)

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties using Kotest framework
- Unit tests validate specific examples and edge cases
- The implementation leverages existing SKaiNET components to minimize development effort
- IREE integration is documented as future work; current implementation uses CPU backend with StableHLO compilation preparation