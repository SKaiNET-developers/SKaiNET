package sk.ainet.data.fashionmnist

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
 * JVM implementation of the Fashion-MNIST loader.
 *
 * @property config The configuration for the Fashion-MNIST loader.
 */
public class FashionMNISTLoaderJvm(config: FashionMNISTLoaderConfig) : FashionMNISTLoaderCommon(config) {

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
            println("Downloading Fashion-MNIST file: $url")
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
                requestTimeoutMillis = 60000 // 60 seconds
                connectTimeoutMillis = 60000 // 60 seconds
                socketTimeoutMillis = 60000 // 60 seconds
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
         * Creates a new instance of FashionMNISTLoaderJvm with the default configuration.
         *
         * @return A new instance of FashionMNISTLoaderJvm.
         */
        public fun create(): FashionMNISTLoaderJvm {
            return FashionMNISTLoaderJvm(FashionMNISTLoaderConfig())
        }

        /**
         * Creates a new instance of FashionMNISTLoaderJvm with a custom cache directory.
         *
         * @param cacheDir The directory to use for caching.
         * @return A new instance of FashionMNISTLoaderJvm.
         */
        public fun create(cacheDir: String): FashionMNISTLoaderJvm {
            return FashionMNISTLoaderJvm(FashionMNISTLoaderConfig(cacheDir = cacheDir))
        }

        /**
         * Creates a new instance of FashionMNISTLoaderJvm with a custom configuration.
         *
         * @param config The configuration to use.
         * @return A new instance of FashionMNISTLoaderJvm.
         */
        public fun create(config: FashionMNISTLoaderConfig): FashionMNISTLoaderJvm {
            return FashionMNISTLoaderJvm(config)
        }
    }
}
