package sk.ainet.data.source

/**
 * Fetches a remote URI into memory. Kept injectable so tests and applications
 * can provide their own HTTP stack or policy layer.
 */
public fun interface RemoteDataSourceFetcher {
    public suspend fun fetch(uri: String, headers: Map<String, String>): ByteArray
}

/**
 * Adds platform or application-specific headers to a resolved remote request.
 */
public fun interface DataSourceHeaderProvider {
    public fun headers(request: DataSourceRequest, parsedUri: ParsedDataSourceUri): Map<String, String>
}

/**
 * Computes checksums for integrity verification without tying resolver policy
 * to a concrete platform crypto API.
 */
public fun interface DataSourceChecksum {
    public fun sha256Hex(bytes: ByteArray): String
}

/**
 * Platform storage adapter used by [DefaultDataSourceResolver].
 */
public interface DataSourceByteStore {
    public suspend fun readLocal(path: String): DataSourceStoredArtifact?
    public suspend fun readCache(cacheKey: String): DataSourceStoredArtifact?
    public suspend fun writeCache(cacheKey: String, bytes: ByteArray): DataSourceStoredArtifact
}

/**
 * A platform materialized artifact used by the common resolver core.
 */
public class DataSourceStoredArtifact(
    public val localPath: String?,
    public val sizeBytes: Long?,
    private val byteReader: suspend () -> ByteArray
) {
    public suspend fun readBytes(): ByteArray = byteReader()
}

/**
 * Platform-neutral resolver implementation for local files, HTTP(S), and
 * Hugging Face source URIs. Storage, network, auth, and checksum details are
 * injected so this policy can be reused by each KMP target.
 */
public class DefaultDataSourceResolver(
    private val store: DataSourceByteStore,
    private val fetcher: RemoteDataSourceFetcher,
    private val checksum: DataSourceChecksum,
    private val headerProvider: DataSourceHeaderProvider = DataSourceHeaderProvider { request, _ ->
        request.headers
    }
) : DataSourceResolver {
    override suspend fun resolve(request: DataSourceRequest): DataSourceArtifact {
        val parsed = DataSourceUriParser.parse(request.uri)
        return when (parsed.provider) {
            DataSourceProvider.File -> resolveFile(request, parsed)
            DataSourceProvider.Http, DataSourceProvider.HuggingFace -> resolveRemote(request, parsed)
        }
    }

    private suspend fun resolveFile(
        request: DataSourceRequest,
        parsed: ParsedDataSourceUri
    ): DataSourceArtifact {
        val path = parsed.localPath ?: throw DataSourceException("File source has no local path: ${request.uri}")
        val stored = store.readLocal(path)
            ?: throw DataSourceException("Data source file not found: $path")
        request.expectedSha256?.let { verifySha256(stored.readBytes(), it, request.uri) }
        return stored.toArtifact(request, parsed, cacheHit = true)
    }

    private suspend fun resolveRemote(
        request: DataSourceRequest,
        parsed: ParsedDataSourceUri
    ): DataSourceArtifact {
        val canUseCache = request.cachePolicy == CachePolicy.Use || request.cachePolicy == CachePolicy.Offline
        if (canUseCache) {
            val cached = store.readCache(parsed.cacheKey)
            if (cached != null) {
                request.expectedSha256?.let { verifySha256(cached.readBytes(), it, request.uri) }
                return cached.toArtifact(request, parsed, cacheHit = true)
            }
        }

        if (request.cachePolicy == CachePolicy.Offline) {
            throw DataSourceException("No cached artifact available for offline source: ${request.uri}")
        }

        val bytes = fetcher.fetch(parsed.transportUri, headerProvider.headers(request, parsed))
        request.expectedSha256?.let { verifySha256(bytes, it, request.uri) }

        if (request.cachePolicy == CachePolicy.Bypass) {
            return DataSourceArtifact(
                request = request,
                parsedUri = parsed,
                filename = parsed.filename,
                localPath = null,
                sizeBytes = bytes.size.toLong(),
                cacheHit = false,
                byteReader = { bytes }
            )
        }

        val stored = store.writeCache(parsed.cacheKey, bytes)
        return stored.toArtifact(request, parsed, cacheHit = false)
    }

    private suspend fun DataSourceStoredArtifact.toArtifact(
        request: DataSourceRequest,
        parsed: ParsedDataSourceUri,
        cacheHit: Boolean
    ): DataSourceArtifact {
        return DataSourceArtifact(
            request = request,
            parsedUri = parsed,
            filename = parsed.filename,
            localPath = localPath,
            sizeBytes = sizeBytes,
            cacheHit = cacheHit,
            byteReader = { readBytes() }
        )
    }

    private fun verifySha256(bytes: ByteArray, expected: String, uri: String) {
        val actual = checksum.sha256Hex(bytes)
        if (!actual.equals(expected, ignoreCase = true)) {
            throw DataSourceException(
                "SHA-256 mismatch for $uri: expected ${expected.lowercase()}, actual $actual"
            )
        }
    }
}
