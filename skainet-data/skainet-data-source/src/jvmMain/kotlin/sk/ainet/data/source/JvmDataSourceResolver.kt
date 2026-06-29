package sk.ainet.data.source

import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.request.get
import io.ktor.client.request.header
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.security.MessageDigest

/**
 * Fetches a remote URI into memory. Kept injectable so tests and applications
 * can provide their own HTTP stack or policy layer.
 */
public fun interface RemoteDataSourceFetcher {
    public suspend fun fetch(uri: String, headers: Map<String, String>): ByteArray
}

/**
 * Ktor/CIO-backed remote fetcher for JVM data artifacts.
 */
public class KtorRemoteDataSourceFetcher(
    private val client: HttpClient = HttpClient(CIO) {
        expectSuccess = true
        install(HttpTimeout) {
            requestTimeoutMillis = 600_000
            connectTimeoutMillis = 60_000
            socketTimeoutMillis = 600_000
        }
    }
) : RemoteDataSourceFetcher, AutoCloseable {
    override suspend fun fetch(uri: String, headers: Map<String, String>): ByteArray {
        return client.get(uri) {
            headers.forEach { (name, value) -> header(name, value) }
        }.body()
    }

    override fun close() {
        client.close()
    }
}

/**
 * JVM resolver for local files and cached remote artifacts.
 */
public class JvmDataSourceResolver(
    private val cacheDir: File = defaultCacheDir(),
    private val fetcher: RemoteDataSourceFetcher = KtorRemoteDataSourceFetcher()
) : DataSourceResolver {
    override suspend fun resolve(request: DataSourceRequest): DataSourceArtifact = withContext(Dispatchers.IO) {
        val parsed = DataSourceUriParser.parse(request.uri)
        when (parsed.provider) {
            DataSourceProvider.File -> resolveFile(request, parsed)
            DataSourceProvider.Http, DataSourceProvider.HuggingFace -> resolveRemote(request, parsed)
        }
    }

    private fun resolveFile(
        request: DataSourceRequest,
        parsed: ParsedDataSourceUri
    ): DataSourceArtifact {
        val path = parsed.localPath ?: throw DataSourceException("File source has no local path: ${request.uri}")
        val file = File(path)
        require(file.exists()) { "Data source file not found: ${file.absolutePath}" }
        require(file.isFile) { "Data source path is not a file: ${file.absolutePath}" }
        request.expectedSha256?.let { verifySha256(file.readBytes(), it, request.uri) }
        return DataSourceArtifact(
            request = request,
            parsedUri = parsed,
            filename = parsed.filename,
            localPath = file.absolutePath,
            sizeBytes = file.length(),
            cacheHit = true,
            byteReader = { file.readBytes() }
        )
    }

    private suspend fun resolveRemote(
        request: DataSourceRequest,
        parsed: ParsedDataSourceUri
    ): DataSourceArtifact {
        val target = File(cacheDir, parsed.cacheKey)
        val canUseCache = request.cachePolicy == CachePolicy.Use || request.cachePolicy == CachePolicy.Offline
        if (canUseCache && target.exists() && target.isFile) {
            request.expectedSha256?.let { verifySha256(target.readBytes(), it, request.uri) }
            return cachedArtifact(request, parsed, target, cacheHit = true)
        }

        if (request.cachePolicy == CachePolicy.Offline) {
            throw DataSourceException("No cached artifact available for offline source: ${request.uri}")
        }

        val bytes = fetcher.fetch(parsed.transportUri, requestHeaders(request, parsed))
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

        cacheDir.mkdirs()
        val temp = File(cacheDir, "${parsed.cacheKey}.tmp")
        temp.writeBytes(bytes)
        if (!temp.renameTo(target)) {
            temp.copyTo(target, overwrite = true)
            temp.delete()
        }
        return cachedArtifact(request, parsed, target, cacheHit = false)
    }

    private fun cachedArtifact(
        request: DataSourceRequest,
        parsed: ParsedDataSourceUri,
        target: File,
        cacheHit: Boolean
    ): DataSourceArtifact {
        return DataSourceArtifact(
            request = request,
            parsedUri = parsed,
            filename = parsed.filename,
            localPath = target.absolutePath,
            sizeBytes = target.length(),
            cacheHit = cacheHit,
            byteReader = { target.readBytes() }
        )
    }

    private fun requestHeaders(
        request: DataSourceRequest,
        parsed: ParsedDataSourceUri
    ): Map<String, String> {
        if (parsed.provider != DataSourceProvider.HuggingFace) return request.headers
        if (request.headers.keys.any { it.equals("Authorization", ignoreCase = true) }) return request.headers
        val token = System.getenv("HF_TOKEN")
            ?.takeIf { it.isNotBlank() }
            ?: System.getenv("HUGGING_FACE_HUB_TOKEN")?.takeIf { it.isNotBlank() }
            ?: return request.headers
        return request.headers + ("Authorization" to "Bearer $token")
    }

    private fun verifySha256(bytes: ByteArray, expected: String, uri: String) {
        val actual = MessageDigest.getInstance("SHA-256")
            .digest(bytes)
            .joinToString("") { byte -> "%02x".format(byte) }
        if (!actual.equals(expected, ignoreCase = true)) {
            throw DataSourceException(
                "SHA-256 mismatch for $uri: expected ${expected.lowercase()}, actual $actual"
            )
        }
    }

    public companion object {
        public fun defaultCacheDir(): File {
            val userHome = System.getProperty("user.home")?.takeIf { it.isNotBlank() }
            val base = userHome ?: System.getProperty("java.io.tmpdir")
            return File(base, ".cache/skainet/data")
        }
    }
}
