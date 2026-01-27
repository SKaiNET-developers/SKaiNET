package sk.ainet.data.mnist

import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.logging.Logging
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.request.get
import io.ktor.client.statement.HttpResponse
import io.ktor.client.call.body
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.zip.GZIPInputStream

/**
 * JVM implementation of the MNIST loader.
 *
 * @property config The configuration for the MNIST loader.
 */
public class MNISTLoaderJvm(config: MNISTLoaderConfig) : MNISTLoaderCommon(config) {

    /**
     * Downloads and caches a file.
     *
     * @param url The URL to download from.
     * @param filename The name of the file to save.
     * @return The bytes of the decompressed file.
     */
    override suspend fun downloadAndCacheFile(url: String, filename: String): ByteArray = withContext(Dispatchers.IO) {
        val cacheDir = File(config.cacheDir)
        if (!cacheDir.exists()) {
            cacheDir.mkdirs()
        }

        val gzipFile = File(cacheDir, filename)
        val decompressedFile = File(cacheDir, filename.removeSuffix(".gz"))

        // Check if the decompressed file already exists in cache
        if (config.useCache && decompressedFile.exists()) {
            println("Using cached file: ${decompressedFile.path}")
            return@withContext decompressedFile.readBytes()
        }

        // Check if the gzip file already exists in cache
        if (!gzipFile.exists() || !config.useCache) {
            println("Downloading file: $url")
            downloadFile(url, gzipFile.path)
        } else {
            println("Using cached gzip file: ${gzipFile.path}")
        }

        // Decompress the gzip file
        println("Decompressing file: ${gzipFile.path}")
        decompressGzipFile(gzipFile.path, decompressedFile.path)

        return@withContext decompressedFile.readBytes()
    }

    /**
     * Downloads a file from a URL.
     *
     * @param url The URL to download from.
     * @param outputPath The path to save the file to.
     */
    private suspend fun downloadFile(url: String, outputPath: String) {
        val client = HttpClient(CIO) {
            install(Logging)

            // Configure timeout for large files
            install(HttpTimeout) {
                requestTimeoutMillis = 300000 // 5 minutes
                connectTimeoutMillis = 60000 // 60 seconds
                socketTimeoutMillis = 300000 // 5 minutes
            }
        }

        try {
            val file = File(outputPath)

            val httpResponse: HttpResponse = client.get(url)
            val responseBody: ByteArray = httpResponse.body()
            file.writeBytes(responseBody)

            println("File saved to ${file.path}")
        } finally {
            client.close()
        }
    }

    /**
     * Decompresses a gzip file.
     *
     * @param gzipFilePath The path to the gzip file.
     * @param outputFilePath The path to save the decompressed file to.
     */
    private fun decompressGzipFile(gzipFilePath: String, outputFilePath: String) {
        GZIPInputStream(FileInputStream(gzipFilePath)).use { gzipInputStream ->
            FileOutputStream(outputFilePath).use { outputStream ->
                val buffer = ByteArray(1024)
                var len: Int
                while (gzipInputStream.read(buffer).also { len = it } > 0) {
                    outputStream.write(buffer, 0, len)
                }
            }
        }
    }

    public companion object {
        /**
         * Creates a new instance of MNISTLoaderJvm with the default configuration.
         *
         * @return A new instance of MNISTLoaderJvm.
         */
        public fun create(): MNISTLoaderJvm {
            return MNISTLoaderJvm(MNISTLoaderConfig())
        }

        /**
         * Creates a new instance of MNISTLoaderJvm with a custom cache directory.
         *
         * @param cacheDir The directory to use for caching.
         * @return A new instance of MNISTLoaderJvm.
         */
        public fun create(cacheDir: String): MNISTLoaderJvm {
            return MNISTLoaderJvm(MNISTLoaderConfig(cacheDir = cacheDir))
        }

        /**
         * Creates a new instance of MNISTLoaderJvm with a custom configuration.
         *
         * @param config The configuration to use.
         * @return A new instance of MNISTLoaderJvm.
         */
        public fun create(config: MNISTLoaderConfig): MNISTLoaderJvm {
            return MNISTLoaderJvm(config)
        }
    }
}
