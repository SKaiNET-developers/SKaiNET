# Performance Guide and Optimization

This document provides detailed guidance on optimizing performance and troubleshooting issues with the SKaiNET Grayscale Image CLI.

## Model Selection for Performance

### RGB2GRAYSCALE (Standard Model)
- **Optimized for**: CPU execution, memory efficiency
- **Best use cases**: 
  - Development environments without GPU
  - Systems with limited memory
  - Single image processing
  - Jetson Nano with memory constraints
- **Performance characteristics**:
  - Lower memory footprint (~150MB for 1920x1080 images)
  - Consistent performance across platforms
  - No GPU driver dependencies

### RGB2GRAYSCALE_MATMUL (GPU-Optimized Model)
- **Optimized for**: GPU acceleration, batch processing
- **Best use cases**:
  - NVIDIA Jetson Xavier/Orin devices
  - Systems with CUDA-compatible GPUs
  - Large batch processing workflows
  - High-throughput applications
- **Performance characteristics**:
  - Higher memory usage (~200MB for 1920x1080 images)
  - 2-5x speedup on GPU-enabled systems
  - Requires CUDA drivers and compatible hardware

## Hardware-Specific Optimization

### NVIDIA Jetson Devices

#### Jetson Nano (4GB)
```bash
# Recommended configuration for memory efficiency
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input photos/ --batch --model RGB2GRAYSCALE --verbose"

# For better performance with sufficient memory
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input photos/ --batch --model RGB2GRAYSCALE_MATMUL --verbose"
```

**Optimization tips:**
- Use RGB2GRAYSCALE for large images (>4K) to avoid memory issues
- Process smaller batches (10-20 images) to prevent memory exhaustion
- Monitor memory usage with `--verbose` flag

#### Jetson Xavier/Orin
```bash
# Optimal configuration for maximum performance
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input photos/ --batch --model RGB2GRAYSCALE_MATMUL --verbose"
```

**Optimization tips:**
- Always use RGB2GRAYSCALE_MATMUL for best GPU utilization
- Can handle larger batches (50-100 images)
- Enable verbose mode to verify GPU acceleration is active

### Desktop/Server Systems

#### With NVIDIA GPU
```bash
# High-performance configuration
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input large_dataset/ --output processed/ --batch --model RGB2GRAYSCALE_MATMUL --verbose"
```

#### CPU-Only Systems
```bash
# CPU-optimized configuration
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input photos/ --batch --model RGB2GRAYSCALE --verbose"
```

## Performance Monitoring

### Using Verbose Mode
The `--verbose` flag provides detailed performance information:

```bash
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input test.jpg --model RGB2GRAYSCALE_MATMUL --verbose"
```

**Verbose output includes:**
- System validation results
- GPU detection and capabilities
- CUDA driver version and status
- Memory usage patterns
- Processing time breakdown
- Model compilation status

### Performance Metrics

#### Key Timing Metrics
- **Image Loading**: Time to load and decode image files
- **Tensor Conversion**: Time to convert images to tensor format
- **Model Compilation**: Time to compile model to StableHLO/IREE
- **Execution**: Time for actual grayscale conversion
- **Image Saving**: Time to encode and save output images

#### Memory Metrics
- **Peak Memory Usage**: Maximum memory consumption during processing
- **GPU Memory**: VRAM usage on GPU-enabled systems
- **Tensor Memory**: Memory used for tensor operations

## Troubleshooting Performance Issues

### GPU Not Being Used

**Symptoms:**
- Processing times similar to CPU-only systems
- Verbose output shows "GPU: false"
- No CUDA-related messages in verbose output

**Diagnosis:**
```bash
# Check system capabilities
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input test.jpg --verbose"
```

**Solutions:**
1. **Install CUDA Drivers:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install cuda-drivers
   
   # Verify installation
   nvidia-smi
   ```

2. **Check GPU Compatibility:**
   - Ensure GPU has Compute Capability 3.5 or higher
   - Verify CUDA drivers are version 11.0 or higher

3. **Use GPU-Optimized Model:**
   ```bash
   ./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input test.jpg --model RGB2GRAYSCALE_MATMUL --verbose"
   ```

### Memory Issues

**Symptoms:**
- Out of memory errors
- System becomes unresponsive
- Processing fails on large images

**Solutions:**
1. **Use Memory-Efficient Model:**
   ```bash
   ./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input large_image.jpg --model RGB2GRAYSCALE"
   ```

2. **Process Smaller Batches:**
   ```bash
   # Instead of processing entire directory at once, process in chunks
   ./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input subset1/ --batch --model RGB2GRAYSCALE"
   ./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input subset2/ --batch --model RGB2GRAYSCALE"
   ```

3. **Increase JVM Memory:**
   ```bash
   export GRADLE_OPTS="-Xmx4g"
   ./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input photos/ --batch"
   ```

### Slow Processing

**Symptoms:**
- Processing times much slower than expected
- High CPU usage but low GPU usage
- Batch processing taking excessive time

**Diagnosis Steps:**
1. **Check Hardware Utilization:**
   ```bash
   # Monitor during processing
   nvidia-smi  # For GPU usage
   htop        # For CPU/memory usage
   ```

2. **Verify Model Selection:**
   ```bash
   # Test both models with verbose output
   ./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input test.jpg --model RGB2GRAYSCALE --verbose"
   ./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input test.jpg --model RGB2GRAYSCALE_MATMUL --verbose"
   ```

**Solutions:**
1. **Use Appropriate Model for Hardware:**
   - GPU systems: Use RGB2GRAYSCALE_MATMUL
   - CPU systems: Use RGB2GRAYSCALE

2. **Optimize Batch Size:**
   - Start with smaller batches and increase gradually
   - Monitor memory usage to find optimal batch size

3. **Check System Resources:**
   - Ensure sufficient RAM available
   - Close unnecessary applications
   - Check disk I/O performance

## System Requirements and Dependencies

### Minimum Requirements
- **CPU**: 2+ cores, 2GHz+
- **RAM**: 2GB minimum, 4GB recommended
- **Storage**: 1GB free space for temporary files
- **Java**: JDK 21 or higher

### Recommended Requirements
- **CPU**: 4+ cores, 3GHz+
- **RAM**: 8GB or higher
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: SSD for better I/O performance

### Dependency Validation

The application performs automatic dependency validation when using `--verbose` mode:

```bash
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input test.jpg --verbose"
```

**Validation includes:**
- Java version and compatibility
- CUDA driver availability and version
- GPU capabilities and memory
- SKaiNET module versions
- System memory and storage

### Missing Dependency Resolution

#### Java Version Issues
**Error:** "Unsupported Java version"
**Solution:**
```bash
# Check current Java version
java -version

# Install Java 21 (Ubuntu/Debian)
sudo apt-get install openjdk-21-jdk

# Set JAVA_HOME if needed
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
```

#### CUDA Driver Issues
**Error:** "CUDA drivers not found" or "GPU not available"
**Solution:**
```bash
# Check NVIDIA driver status
nvidia-smi

# Install CUDA drivers (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install cuda-drivers

# For Jetson devices
sudo apt-get install nvidia-jetpack
```

#### Memory Issues
**Error:** "OutOfMemoryError" or "Insufficient memory"
**Solution:**
```bash
# Increase JVM heap size
export GRADLE_OPTS="-Xmx8g"

# Or use memory-efficient model
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input photos/ --model RGB2GRAYSCALE"
```

## Performance Benchmarking

### Benchmark Script
```bash
#!/bin/bash
# benchmark.sh - Performance testing script

echo "SKaiNET Grayscale CLI Performance Benchmark"
echo "==========================================="

# Test single image processing
echo "Testing single image processing..."
time ./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input test.jpg --model RGB2GRAYSCALE"
time ./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input test.jpg --model RGB2GRAYSCALE_MATMUL"

# Test batch processing (if test directory exists)
if [ -d "test_images" ]; then
    echo "Testing batch processing..."
    time ./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input test_images/ --batch --model RGB2GRAYSCALE"
    time ./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input test_images/ --batch --model RGB2GRAYSCALE_MATMUL"
fi

echo "Benchmark completed!"
```

### Expected Performance Ranges

| Hardware Configuration | Single Image (1920x1080) | Batch Processing (images/sec) |
|------------------------|---------------------------|-------------------------------|
| Jetson Nano + RGB2GRAYSCALE | 150-250ms | 40-60 |
| Jetson Nano + RGB2GRAYSCALE_MATMUL | 120-200ms | 50-70 |
| Jetson Xavier + RGB2GRAYSCALE | 80-150ms | 60-90 |
| Jetson Xavier + RGB2GRAYSCALE_MATMUL | 50-100ms | 100-150 |
| Desktop CPU + RGB2GRAYSCALE | 100-200ms | 50-80 |
| Desktop GPU + RGB2GRAYSCALE_MATMUL | 30-80ms | 150-300 |

### Performance Optimization Checklist

- [ ] Use appropriate model for hardware (GPU vs CPU)
- [ ] Enable verbose mode to verify GPU utilization
- [ ] Monitor memory usage and adjust batch sizes
- [ ] Ensure CUDA drivers are installed and up-to-date
- [ ] Use SSD storage for better I/O performance
- [ ] Close unnecessary applications to free resources
- [ ] Consider image resolution vs processing time trade-offs
- [ ] Test both models to find optimal performance for your use case

## Advanced Performance Tuning

### JVM Tuning
```bash
# Optimize garbage collection for large batches
export GRADLE_OPTS="-Xmx8g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"

# For systems with limited memory
export GRADLE_OPTS="-Xmx2g -XX:+UseSerialGC"
```

### GPU Memory Management
```bash
# Monitor GPU memory usage during processing
watch -n 1 nvidia-smi

# For memory-constrained GPUs, use smaller batches
./gradlew :skainet-apps:skainet-grayscale-cli:run --args="--input small_batch/ --batch --model RGB2GRAYSCALE_MATMUL"
```

### I/O Optimization
- Use local storage instead of network drives
- Prefer SSD over HDD for temporary files
- Ensure sufficient disk space for output files
- Consider parallel processing for very large datasets

This performance guide should help you achieve optimal results with the SKaiNET Grayscale Image CLI across different hardware configurations and use cases.