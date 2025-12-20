# SKaiNET Grayscale Image CLI

A high-performance command-line interface application for converting color images to grayscale using SKaiNET's neural network capabilities with StableHLO compilation. Designed for deployment on NVIDIA Jetson devices and development environments with automatic GPU/CPU fallback.

## Features

- **Neural Network Conversion**: Uses SKaiNET's Rgb2GrayScale and Rgb2GrayScaleMatMul models for accurate grayscale conversion
- **StableHLO Compilation**: Compiles models to optimized MLIR format for maximum performance
- **GPU Acceleration**: Automatic GPU detection with CPU fallback for broad compatibility
- **Batch Processing**: Process entire directories while preserving folder structure
- **Format Support**: JPEG, PNG, BMP, and GIF input/output with format preservation
- **Error Resilience**: Continues processing remaining images when individual files fail
- **Comprehensive Logging**: Detailed progress reporting and performance metrics

## Installation

### Prerequisites

#### System Requirements
- **Java**: JDK 21 or higher (enforced by Gradle toolchain)
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 2GB RAM (4GB+ recommended for large images)

#### For GPU Acceleration (Optional)
- **NVIDIA GPU**: CUDA-compatible GPU (Compute Capability 3.5+)
- **CUDA Drivers**: NVIDIA CUDA drivers 11.0 or higher
- **Jetson Devices**: JetPack 4.6+ for Jetson Nano/Xavier/Orin

#### CUDA Driver Installation

**Ubuntu/Debian:**
```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA drivers
sudo apt-get install cuda-drivers

# Verify installation
nvidia-smi
```

**NVIDIA Jetson:**
```bash
# Install JetPack (includes CUDA drivers)
sudo apt update
sudo apt install nvidia-jetpack

# Verify installation
nvcc --version
```

**Other Systems:**
- Download drivers from [NVIDIA Driver Downloads](https://www.nvidia.com/drivers)
- Follow platform-specific installation instructions

### Building the Application

1. **Clone the SKaiNET repository:**
   ```bash
   git clone https://github.com/your-org/skainet.git
   cd skainet
   ```

2. **Build the application:**
   ```bash
   ./gradlew :skainet-apps:skainet-grayscale-cli:build
   ```

3. **Verify installation:**
   ```bash
   ./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--help"
   ```

## Usage

### Command-Line Options

```
Usage: grayscale-cli [OPTIONS]

Options:
  -i, --input PATH        Input image file path or directory (required)
  -o, --output PATH       Output image file path or directory (optional)
  -m, --model MODEL       Grayscale conversion model (default: RGB2GRAYSCALE)
  -b, --batch             Enable batch processing for directories
  -v, --verbose           Enable verbose output and system validation
  -h, --help              Show this help message and exit

Available Models:
  RGB2GRAYSCALE          Standard RGB to grayscale conversion using luminance weights
  RGB2GRAYSCALE_MATMUL   Matrix multiplication-based conversion (optimized for GPU)
```

### Single Image Processing

#### Basic Usage
```bash
# Convert a single image (auto-generates output filename)
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input photo.jpg"
# Output: photo_gray.jpg

# Specify custom output path
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input photo.jpg --output grayscale_photo.jpg"
```

#### Model Selection
```bash
# Use standard model (default)
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input photo.jpg --model RGB2GRAYSCALE"

# Use matrix multiplication model (better for GPU)
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input photo.jpg --model RGB2GRAYSCALE_MATMUL"
```

#### Verbose Mode
```bash
# Enable detailed logging and system validation
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input photo.jpg --verbose"
```

### Batch Processing

#### Process Directory
```bash
# Process all images in a directory
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input /path/to/images --batch"
# Output directory: /path/to/images_gray/

# Specify custom output directory
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input /path/to/images --output /path/to/output --batch"
```

#### Batch Processing Features
- **Recursive Processing**: Processes all subdirectories
- **Structure Preservation**: Maintains original folder hierarchy
- **Format Detection**: Automatically detects supported image formats
- **Error Resilience**: Continues processing when individual images fail
- **Progress Reporting**: Shows current file and completion percentage

### Advanced Examples

#### High-Performance GPU Processing
```bash
# Use GPU-optimized model with verbose logging
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input large_image.png --model RGB2GRAYSCALE_MATMUL --verbose"
```

#### Batch Processing with Custom Model
```bash
# Process directory with matrix multiplication model
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input /photos --output /processed --batch --model RGB2GRAYSCALE_MATMUL --verbose"
```

## Model Selection Guide

### RGB2GRAYSCALE (Default)
- **Best for**: General-purpose conversion, CPU processing
- **Performance**: Optimized for CPU execution
- **Accuracy**: Standard luminance-based conversion (0.299R + 0.587G + 0.114B)
- **Memory**: Lower memory usage
- **Recommended**: Development environments, CPU-only systems

### RGB2GRAYSCALE_MATMUL
- **Best for**: GPU acceleration, batch processing
- **Performance**: Optimized for parallel execution on GPU
- **Accuracy**: Identical to RGB2GRAYSCALE but computed via matrix operations
- **Memory**: Higher memory usage but better throughput
- **Recommended**: NVIDIA Jetson devices, systems with CUDA support

## Performance Considerations

### Hardware Optimization

#### NVIDIA Jetson Devices
- **Jetson Nano**: Use RGB2GRAYSCALE for memory efficiency
- **Jetson Xavier/Orin**: Use RGB2GRAYSCALE_MATMUL for maximum performance
- **Batch Size**: Process 10-50 images per batch for optimal memory usage

#### Development Machines
- **With GPU**: Use RGB2GRAYSCALE_MATMUL with verbose mode to verify GPU usage
- **CPU Only**: Use RGB2GRAYSCALE for best performance
- **Large Images**: Monitor memory usage with verbose logging

### Performance Tips

1. **GPU Utilization**: Use `--verbose` to verify GPU acceleration is active
2. **Batch Processing**: More efficient than processing individual files
3. **Model Selection**: Choose RGB2GRAYSCALE_MATMUL for GPU systems
4. **Memory Management**: Process large batches in smaller chunks if memory is limited

### Expected Performance
- **Single Image (1920x1080)**: 50-200ms depending on hardware
- **Batch Processing**: 10-100 images/second depending on size and hardware
- **GPU Acceleration**: 2-5x speedup over CPU on supported hardware

## Troubleshooting

### Common Issues

#### "CUDA drivers not found"
**Solution:**
1. Install NVIDIA CUDA drivers (see Installation section)
2. Verify with `nvidia-smi` command
3. Use `--verbose` flag to see detailed system validation
4. Application will automatically fall back to CPU if GPU unavailable

#### "Input file not found"
**Solution:**
1. Verify file path is correct and file exists
2. Check file permissions (read access required)
3. Ensure file extension is supported (.jpg, .jpeg, .png, .bmp, .gif)

#### "Permission denied" errors
**Solution:**
1. Check write permissions for output directory
2. Ensure sufficient disk space
3. Run with appropriate user permissions

#### "Unsupported image format"
**Solution:**
1. Convert image to supported format (JPEG, PNG, BMP, GIF)
2. Check file is not corrupted
3. Use `--verbose` for detailed error information

#### Out of memory errors
**Solution:**
1. Use RGB2GRAYSCALE model for lower memory usage
2. Process smaller batches
3. Reduce image resolution before processing
4. Ensure sufficient system RAM

### System Validation

Use the `--verbose` flag to perform comprehensive system validation:

```bash
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input test.jpg --verbose"
```

This will report:
- CUDA driver availability
- GPU memory and capabilities
- SKaiNET module versions
- System performance characteristics

### Getting Help

1. **Command Help**: Use `--help` flag for usage information
2. **Verbose Logging**: Use `--verbose` for detailed execution information
3. **Error Messages**: Application provides specific error details and suggestions
4. **System Validation**: Verbose mode includes comprehensive dependency checking

## Dependencies

### SKaiNET Components
- **skainet-lang-models**: Grayscale conversion model definitions
- **skainet-io-image**: Image loading and saving functionality
- **skainet-compile-hlo**: StableHLO compilation for GPU optimization
- **skainet-backend-cpu**: CPU execution backend with fallback support

### External Dependencies
- **Kotlin**: 2.2.21 (Multiplatform framework)
- **kotlinx-cli**: Command-line argument parsing
- **kotlinx-coroutines**: Asynchronous processing
- **kotlinx-serialization**: Configuration and metadata handling

### Runtime Requirements
- **JVM**: Java 21+ (automatically managed by Gradle toolchain)
- **Native Libraries**: Platform-specific image processing libraries
- **CUDA Runtime**: Optional, for GPU acceleration

## Development and Contributing

### Building from Source
```bash
# Full build with tests
./gradlew :skainet-apps:skainet-grayscale-cli:build

# Run tests only
./gradlew :skainet-apps:skainet-grayscale-cli:test

# Generate coverage report
./gradlew :skainet-apps:skainet-grayscale-cli:koverHtmlReport
```

### Project Structure
```
skainet-apps/skainet-grayscale-cli/
├── src/main/kotlin/sk/ainet/apps/grayscale/
│   ├── GrayscaleImageCli.kt          # Main CLI application
│   ├── ImageProcessingPipeline.kt    # Core processing logic
│   ├── ImageLoader.kt                # Image I/O operations
│   ├── HloCompiler.kt               # StableHLO compilation
│   └── ErrorHandler.kt              # Error handling and reporting
├── src/test/kotlin/                 # Unit and property tests
├── build.gradle.kts                 # Build configuration
└── README.md                        # This documentation
```

### Testing
The application includes comprehensive test coverage:
- **Unit Tests**: Individual component testing
- **Property Tests**: Randomized testing with Kotest
- **Integration Tests**: End-to-end workflow validation

## License

This project is part of the SKaiNET framework. See the main repository for license information.