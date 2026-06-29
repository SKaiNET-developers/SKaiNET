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
    cacheDir: File = defaultCacheDir(),
    fetcher: RemoteDataSourceFetcher = KtorRemoteDataSourceFetcher()
) : DataSourceResolver {
    private val delegate = DefaultDataSourceResolver(
        store = JvmFileDataSourceByteStore(cacheDir),
        fetcher = fetcher,
        checksum = JvmSha256DataSourceChecksum,
        headerProvider = JvmHuggingFaceHeaderProvider
    )

    override suspend fun resolve(request: DataSourceRequest): DataSourceArtifact = withContext(Dispatchers.IO) {
        delegate.resolve(request)
    }

    public companion object {
        public fun defaultCacheDir(): File {
            val userHome = System.getProperty("user.home")?.takeIf { it.isNotBlank() }
            val base = userHome ?: System.getProperty("java.io.tmpdir")
            return File(base, ".cache/skainet/data")
        }
    }
}

internal class JvmFileDataSourceByteStore(
    private val cacheDir: File
) : DataSourceByteStore {
    override suspend fun readLocal(path: String): DataSourceStoredArtifact? {
        val file = File(path)
        if (!file.exists()) return null
        if (!file.isFile) {
            throw DataSourceException("Data source path is not a file: ${file.absolutePath}")
        }
        return file.toStoredArtifact()
    }

    override suspend fun readCache(cacheKey: String): DataSourceStoredArtifact? {
        val target = File(cacheDir, cacheKey)
        return if (target.exists() && target.isFile) target.toStoredArtifact() else null
    }

    override suspend fun writeCache(cacheKey: String, bytes: ByteArray): DataSourceStoredArtifact {
        cacheDir.mkdirs()
        val target = File(cacheDir, cacheKey)
        val temp = File(cacheDir, "$cacheKey.tmp")
        temp.writeBytes(bytes)
        if (!temp.renameTo(target)) {
            temp.copyTo(target, overwrite = true)
            temp.delete()
        }
        return target.toStoredArtifact()
    }

    private fun File.toStoredArtifact(): DataSourceStoredArtifact {
        return DataSourceStoredArtifact(
            localPath = absolutePath,
            sizeBytes = length(),
            byteReader = { readBytes() }
        )
    }
}

internal object JvmHuggingFaceHeaderProvider : DataSourceHeaderProvider {
    override fun headers(request: DataSourceRequest, parsedUri: ParsedDataSourceUri): Map<String, String> {
        if (parsedUri.provider != DataSourceProvider.HuggingFace) return request.headers
        if (request.headers.keys.any { it.equals("Authorization", ignoreCase = true) }) return request.headers
        val token = System.getenv("HF_TOKEN")
            ?.takeIf { it.isNotBlank() }
            ?: System.getenv("HUGGING_FACE_HUB_TOKEN")?.takeIf { it.isNotBlank() }
            ?: return request.headers
        return request.headers + ("Authorization" to "Bearer $token")
    }
}

internal object JvmSha256DataSourceChecksum : DataSourceChecksum {
    override fun sha256Hex(bytes: ByteArray): String {
        return MessageDigest.getInstance("SHA-256")
            .digest(bytes)
            .joinToString("") { byte -> "%02x".format(byte) }
    }
}
