package sk.ainet.data.source

import kotlinx.io.Sink
import kotlinx.io.Source
import kotlinx.io.readByteArray

/**
 * Cache behavior requested by a caller resolving a data artifact.
 */
public enum class CachePolicy {
    /** Use a cached artifact when present, otherwise fetch and cache it. */
    Use,

    /** Fetch the artifact again and replace any existing cached copy. */
    Refresh,

    /** Require an existing cached or local artifact; do not use the network. */
    Offline,

    /** Fetch or read the artifact without writing a persistent cache entry. */
    Bypass
}

/**
 * High-level provider implied by a source URI.
 */
public enum class DataSourceProvider {
    File,
    Http,
    HuggingFace
}

/**
 * Hugging Face repository namespace encoded by an `hf://` URI.
 */
public enum class HuggingFaceRepoType {
    Model,
    Dataset,
    Space
}

/**
 * Parsed Hugging Face location, when a URI uses SKaiNET's `hf://` shorthand
 * or the explicit `hf+https://...` provider prefix.
 */
public data class HuggingFaceLocation(
    public val repoType: HuggingFaceRepoType,
    public val repoId: String?,
    public val revision: String?,
    public val path: String?
)

/**
 * A normalized, provider-aware source URI.
 */
public data class ParsedDataSourceUri(
    public val rawUri: String,
    public val provider: DataSourceProvider,
    public val transportUri: String,
    public val filename: String,
    public val cacheKey: String,
    public val localPath: String? = null,
    public val huggingFace: HuggingFaceLocation? = null
)

/**
 * Request to resolve a local or remote artifact.
 */
public data class DataSourceRequest(
    public val uri: String,
    public val cachePolicy: CachePolicy = CachePolicy.Use,
    public val expectedSha256: String? = null,
    public val headers: Map<String, String> = emptyMap()
)

/**
 * A resolved artifact. Remote artifacts may expose a [localPath] when they
 * have been materialized into a platform cache.
 */
public class DataSourceArtifact(
    public val request: DataSourceRequest,
    public val parsedUri: ParsedDataSourceUri,
    public val filename: String,
    public val localPath: String?,
    public val sizeBytes: Long?,
    public val cacheHit: Boolean,
    private val sourceOpener: suspend () -> Source
) {
    /**
     * Opens a fresh source for this artifact. Callers own and must close it.
     */
    public suspend fun openSource(): Source = sourceOpener()

    /**
     * Convenience for small artifacts. Prefer [openSource] or [copyTo] for
     * model-scale data.
     */
    public suspend fun readBytes(): ByteArray {
        val source = openSource()
        return try {
            source.readByteArray()
        } finally {
            source.close()
        }
    }

    /**
     * Streams this artifact into [sink]. The source is closed after copying;
     * [sink] is left open for the caller.
     */
    public suspend fun copyTo(sink: Sink): Long {
        val source = openSource()
        return try {
            source.transferTo(sink)
        } finally {
            source.close()
        }
    }
}

/**
 * Resolves source URIs into readable data artifacts.
 */
public interface DataSourceResolver {
    public suspend fun resolve(request: DataSourceRequest): DataSourceArtifact
}

public open class DataSourceException(
    message: String,
    cause: Throwable? = null
) : RuntimeException(message, cause)

public class UnsupportedDataSourceUriException(
    uri: String
) : DataSourceException("Unsupported data source URI: $uri")
