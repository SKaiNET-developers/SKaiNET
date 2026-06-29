package sk.ainet.data.source

import kotlinx.io.Buffer
import kotlinx.io.RawSource
import kotlinx.io.Sink
import kotlinx.io.Source
import kotlinx.io.buffered
import kotlinx.io.files.FileSystem
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readByteArray

/**
 * Remote response body exposed as a one-shot [Source].
 */
public class DataSourceRemoteContent(
    public val source: Source,
    public val sizeBytes: Long? = null
) {
    public companion object {
        public fun fromBytes(bytes: ByteArray): DataSourceRemoteContent {
            return DataSourceRemoteContent(
                source = bytes.toDataSourceSource(),
                sizeBytes = bytes.size.toLong()
            )
        }
    }
}

/**
 * Fetches a remote URI as a stream. Kept injectable so tests and applications
 * can provide their own HTTP stack or policy layer.
 */
public fun interface RemoteDataSourceFetcher {
    public suspend fun fetch(uri: String, headers: Map<String, String>): DataSourceRemoteContent
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
public interface DataSourceChecksum {
    public fun newSha256(): DataSourceHash
}

/**
 * Incremental hash state used while streaming artifact bytes.
 */
public interface DataSourceHash {
    public fun update(bytes: ByteArray, startIndex: Int = 0, endIndex: Int = bytes.size)
    public fun hex(): String
}

/**
 * Platform storage adapter used by [DefaultDataSourceResolver].
 */
public interface DataSourceArtifactStore {
    public suspend fun readLocal(path: String): DataSourceStoredArtifact?
    public suspend fun readCache(cacheKey: String): DataSourceStoredArtifact?
    public suspend fun writeCache(
        cacheKey: String,
        source: Source,
        sizeBytes: Long? = null,
        validate: suspend (DataSourceStoredArtifact) -> Unit = {}
    ): DataSourceStoredArtifact
}

/**
 * A materialized artifact used by the common resolver core.
 */
public class DataSourceStoredArtifact(
    public val localPath: String?,
    public val sizeBytes: Long?,
    private val sourceOpener: suspend () -> Source
) {
    public suspend fun openSource(): Source = sourceOpener()

    public suspend fun readBytes(): ByteArray {
        val source = openSource()
        return try {
            source.readByteArray()
        } finally {
            source.close()
        }
    }

    public suspend fun copyTo(sink: Sink): Long {
        val source = openSource()
        return try {
            source.transferTo(sink)
        } finally {
            source.close()
        }
    }

    public companion object {
        public fun inMemory(
            bytes: ByteArray,
            localPath: String? = null
        ): DataSourceStoredArtifact {
            val owned = bytes.copyOf()
            return DataSourceStoredArtifact(
                localPath = localPath,
                sizeBytes = owned.size.toLong(),
                sourceOpener = { owned.toDataSourceSource() }
            )
        }

        public fun inMemoryFrom(
            source: Source,
            localPath: String? = null,
            sizeBytes: Long? = null
        ): DataSourceStoredArtifact {
            val buffer = Buffer()
            val copied = source.transferTo(buffer)
            return DataSourceStoredArtifact(
                localPath = localPath,
                sizeBytes = sizeBytes ?: copied,
                sourceOpener = { buffer.copy() }
            )
        }
    }
}

/**
 * Filesystem-backed artifact store built on kotlinx-io so the cache policy
 * remains reusable across KMP targets that expose [SystemFileSystem].
 */
public class FileSystemDataSourceArtifactStore(
    private val cacheDir: Path,
    private val fileSystem: FileSystem = SystemFileSystem
) : DataSourceArtifactStore {
    override suspend fun readLocal(path: String): DataSourceStoredArtifact? {
        val localPath = Path(path)
        val metadata = fileSystem.metadataOrNull(localPath) ?: return null
        if (!metadata.isRegularFile) {
            throw DataSourceException("Data source path is not a file: $path")
        }
        val resolved = fileSystem.resolve(localPath)
        return resolved.toStoredArtifact(metadata.size)
    }

    override suspend fun readCache(cacheKey: String): DataSourceStoredArtifact? {
        val target = Path(cacheDir, cacheKey)
        val metadata = fileSystem.metadataOrNull(target) ?: return null
        return if (metadata.isRegularFile) target.toStoredArtifact(metadata.size) else null
    }

    override suspend fun writeCache(
        cacheKey: String,
        source: Source,
        sizeBytes: Long?,
        validate: suspend (DataSourceStoredArtifact) -> Unit
    ): DataSourceStoredArtifact {
        fileSystem.createDirectories(cacheDir)
        val target = Path(cacheDir, cacheKey)
        val temp = Path(cacheDir, "$cacheKey.tmp")

        val sink = fileSystem.sink(temp).buffered()
        try {
            source.transferTo(sink)
            sink.flush()
        } finally {
            sink.close()
        }

        val tempMetadata = fileSystem.metadataOrNull(temp)
        val tempArtifact = temp.toStoredArtifact(tempMetadata?.size ?: sizeBytes)
        try {
            validate(tempArtifact)
        } catch (throwable: Throwable) {
            fileSystem.delete(temp, mustExist = false)
            throw throwable
        }

        fileSystem.atomicMove(temp, target)
        val metadata = fileSystem.metadataOrNull(target)
        return target.toStoredArtifact(metadata?.size ?: sizeBytes)
    }

    private fun Path.toStoredArtifact(sizeBytes: Long?): DataSourceStoredArtifact {
        val path = this
        return DataSourceStoredArtifact(
            localPath = path.toString(),
            sizeBytes = sizeBytes,
            sourceOpener = { fileSystem.source(path).buffered() }
        )
    }
}

/**
 * Platform-neutral resolver implementation for local files, HTTP(S), and
 * Hugging Face source URIs. Storage, network, auth, and checksum details are
 * injected so this policy can be reused by each KMP target.
 */
public class DefaultDataSourceResolver(
    private val store: DataSourceArtifactStore,
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
        request.expectedSha256?.let { verifySha256(stored, it, request.uri) }
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
                request.expectedSha256?.let { verifySha256(cached, it, request.uri) }
                return cached.toArtifact(request, parsed, cacheHit = true)
            }
        }

        if (request.cachePolicy == CachePolicy.Offline) {
            throw DataSourceException("No cached artifact available for offline source: ${request.uri}")
        }

        val remote = fetcher.fetch(parsed.transportUri, headerProvider.headers(request, parsed))

        if (request.cachePolicy == CachePolicy.Bypass) {
            val stored = try {
                DataSourceStoredArtifact.inMemoryFrom(remote.source, sizeBytes = remote.sizeBytes)
            } finally {
                remote.source.close()
            }
            request.expectedSha256?.let { verifySha256(stored, it, request.uri) }
            return stored.toArtifact(request, parsed, cacheHit = false)
        }

        val expectedSha256 = request.expectedSha256
        val hash = expectedSha256?.let { checksum.newSha256() }
        val source = hash?.let { HashingRawSource(remote.source, it).buffered() } ?: remote.source
        val stored = try {
            store.writeCache(
                cacheKey = parsed.cacheKey,
                source = source,
                sizeBytes = remote.sizeBytes,
                validate = {
                    if (expectedSha256 != null && hash != null) {
                        verifySha256Hex(hash.hex(), expectedSha256, request.uri)
                    }
                }
            )
        } finally {
            source.close()
        }
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
            sourceOpener = { openSource() }
        )
    }

    private suspend fun verifySha256(artifact: DataSourceStoredArtifact, expected: String, uri: String) {
        val actual = artifact.sha256Hex()
        verifySha256Hex(actual, expected, uri)
    }

    private fun verifySha256Hex(actual: String, expected: String, uri: String) {
        if (!actual.equals(expected, ignoreCase = true)) {
            throw DataSourceException(
                "SHA-256 mismatch for $uri: expected ${expected.lowercase()}, actual $actual"
            )
        }
    }

    private suspend fun DataSourceStoredArtifact.sha256Hex(): String {
        val hash = checksum.newSha256()
        val buffer = ByteArray(STREAM_BUFFER_SIZE)
        val source = openSource()
        try {
            while (true) {
                val read = source.readAtMostTo(buffer)
                if (read == -1) break
                hash.update(buffer, endIndex = read)
            }
        } finally {
            source.close()
        }
        return hash.hex()
    }

    private companion object {
        private const val STREAM_BUFFER_SIZE = 8 * 1024
    }
}

private class HashingRawSource(
    private val source: Source,
    private val hash: DataSourceHash
) : RawSource {
    override fun readAtMostTo(sink: Buffer, byteCount: Long): Long {
        val start = sink.size
        val read = source.readAtMostTo(sink, byteCount)
        if (read > 0) {
            val copied = Buffer()
            sink.copyTo(copied, startIndex = start, endIndex = start + read)
            hash.update(copied.readByteArray())
        }
        return read
    }

    override fun close() {
        source.close()
    }
}

private fun ByteArray.toDataSourceSource(): Source {
    val buffer = Buffer()
    buffer.write(this)
    return buffer
}
