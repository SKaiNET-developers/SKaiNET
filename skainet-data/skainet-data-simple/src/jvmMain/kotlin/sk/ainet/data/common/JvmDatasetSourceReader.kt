package sk.ainet.data.common

import sk.ainet.data.source.CachePolicy
import sk.ainet.data.source.DataSourceAuthToken
import sk.ainet.data.source.DataSourceRequest
import sk.ainet.data.source.JvmDataSourceResolver
import java.io.ByteArrayInputStream
import java.io.File
import java.util.zip.GZIPInputStream

internal class JvmDatasetSourceReader(
    cacheDir: String,
    useCache: Boolean,
    private val huggingFaceTokenProvider: DatasetHuggingFaceTokenProvider? = null,
    useEnvironmentHuggingFaceToken: Boolean = false
) {
    private val resolver = JvmDataSourceResolver(
        cacheDir = File(cacheDir, "sources"),
        useEnvironmentHuggingFaceToken = useEnvironmentHuggingFaceToken
    )
    private val cachePolicy = if (useCache) CachePolicy.Use else CachePolicy.Refresh

    suspend fun read(uri: String): ByteArray {
        val artifact = resolver.resolve(
            DataSourceRequest(
                uri = uri,
                cachePolicy = cachePolicy,
                huggingFaceToken = DataSourceAuthToken.fromOrNull(huggingFaceTokenProvider?.token())
            )
        )
        return artifact.readBytes()
    }

    suspend fun readGzipDecoded(uri: String): ByteArray = read(uri).gunzipIfNeeded()
}

internal fun ByteArray.gunzip(): ByteArray {
    return GZIPInputStream(ByteArrayInputStream(this)).use { it.readBytes() }
}

internal fun ByteArray.gunzipIfNeeded(): ByteArray {
    return if (isGzip()) gunzip() else this
}

private fun ByteArray.isGzip(): Boolean {
    return size >= 2 && this[0] == 0x1f.toByte() && this[1] == 0x8b.toByte()
}
