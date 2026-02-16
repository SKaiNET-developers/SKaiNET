package sk.ainet.io.gguf.registry

import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader

/**
 * Describes how to create a weight loader for a specific model architecture.
 *
 * Implementations are registered with [ModelLoaderRegistry] and selected
 * automatically based on GGUF metadata.
 */
public interface ModelLoaderDescriptor {
    /** The architecture this descriptor handles. */
    public val architecture: ModelArchitecture

    /** Human-readable name for logging. */
    public val displayName: String
}

/**
 * Registry for model loaders with GGUF auto-detection.
 *
 * Reads the `general.architecture` field from a GGUF file header
 * and resolves the appropriate [ModelLoaderDescriptor].
 *
 * Usage:
 * ```kotlin
 * // Register loaders (typically at startup)
 * ModelLoaderRegistry.register(LlamaLoaderDescriptor)
 * ModelLoaderRegistry.register(Gemma3nLoaderDescriptor)
 *
 * // Auto-detect from file
 * val (arch, descriptor) = ModelLoaderRegistry.detect(source)
 * ```
 */
public object ModelLoaderRegistry {

    private val descriptors = mutableMapOf<ModelArchitecture, ModelLoaderDescriptor>()

    /**
     * Register a loader descriptor for an architecture.
     * Replaces any previously registered descriptor for the same architecture.
     */
    public fun register(descriptor: ModelLoaderDescriptor) {
        descriptors[descriptor.architecture] = descriptor
    }

    /** Return all registered descriptors. */
    public fun registeredDescriptors(): Map<ModelArchitecture, ModelLoaderDescriptor> =
        descriptors.toMap()

    /**
     * Detect the model architecture from a GGUF file.
     *
     * Opens the file, reads only the metadata header (~1MB), extracts
     * `general.architecture`, and resolves to a [ModelArchitecture].
     *
     * @param source Random-access source to the GGUF file
     * @return Detected architecture
     */
    public fun detectArchitecture(source: RandomAccessSource): ModelArchitecture {
        return StreamingGGUFReader.open(source).use { reader ->
            val arch = reader.fields["general.architecture"] as? String
            ModelArchitecture.fromGguf(arch)
        }
    }

    /**
     * Detect the architecture and return the matching descriptor.
     *
     * @param source Random-access source to the GGUF file
     * @return Pair of detected architecture and its descriptor, or null descriptor if not registered
     */
    public fun detect(source: RandomAccessSource): Pair<ModelArchitecture, ModelLoaderDescriptor?> {
        val arch = detectArchitecture(source)
        return arch to descriptors[arch]
    }

    /**
     * Look up the descriptor for a known architecture.
     *
     * @return The registered descriptor, or null if the architecture is not registered
     */
    public fun descriptorFor(architecture: ModelArchitecture): ModelLoaderDescriptor? =
        descriptors[architecture]
}
