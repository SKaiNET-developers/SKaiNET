package sk.ainet.data.source

import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.request.get
import io.ktor.client.request.header
import io.ktor.client.statement.bodyAsChannel
import io.ktor.http.HttpHeaders
import io.ktor.utils.io.asSource
import kotlinx.io.buffered
import kotlinx.io.files.Path
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
    override suspend fun fetch(uri: String, headers: Map<String, String>): DataSourceRemoteContent {
        val response = client.get(uri) {
            headers.forEach { (name, value) -> header(name, value) }
        }
        return DataSourceRemoteContent(
            source = response.bodyAsChannel().asSource().buffered(),
            sizeBytes = response.headers[HttpHeaders.ContentLength]?.toLongOrNull()
        )
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
    fetcher: RemoteDataSourceFetcher = KtorRemoteDataSourceFetcher(),
    huggingFaceToken: DataSourceAuthToken? = null,
    useEnvironmentHuggingFaceToken: Boolean = false
) : DataSourceResolver {
    private val delegate = DefaultDataSourceResolver(
        store = FileSystemDataSourceArtifactStore(Path(cacheDir.absolutePath)),
        fetcher = fetcher,
        checksum = JvmSha256DataSourceChecksum,
        headerProvider = HuggingFaceTokenHeaderProvider(
            JvmHuggingFaceTokenProvider(
                configuredToken = huggingFaceToken,
                useEnvironmentToken = useEnvironmentHuggingFaceToken
            )
        )
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

internal class JvmHuggingFaceTokenProvider(
    private val configuredToken: DataSourceAuthToken?,
    private val useEnvironmentToken: Boolean
) : HuggingFaceTokenProvider {
    override fun token(request: DataSourceRequest, parsedUri: ParsedDataSourceUri): DataSourceAuthToken? {
        if (parsedUri.provider != DataSourceProvider.HuggingFace) return null
        configuredToken?.let { return it }
        if (!useEnvironmentToken) return null
        return DataSourceAuthToken.fromOrNull(System.getenv("HF_TOKEN"))
            ?: DataSourceAuthToken.fromOrNull(System.getenv("HUGGING_FACE_HUB_TOKEN"))
    }
}

internal object JvmSha256DataSourceChecksum : DataSourceChecksum {
    override fun newSha256(): DataSourceHash = JvmSha256DataSourceHash()
}

private class JvmSha256DataSourceHash : DataSourceHash {
    private val digest = MessageDigest.getInstance("SHA-256")

    override fun update(bytes: ByteArray, startIndex: Int, endIndex: Int) {
        digest.update(bytes, startIndex, endIndex - startIndex)
    }

    override fun hex(): String {
        return digest
            .digest()
            .joinToString("") { byte -> "%02x".format(byte) }
    }
}
