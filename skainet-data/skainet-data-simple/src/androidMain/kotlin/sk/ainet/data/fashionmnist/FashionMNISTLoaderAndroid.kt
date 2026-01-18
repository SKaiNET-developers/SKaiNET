package sk.ainet.data.fashionmnist

import io.ktor.client.HttpClient
import io.ktor.client.engine.android.Android
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
 * Android implementation of the Fashion-MNIST loader.
 *
 * @property config The configuration for the Fashion-MNIST loader.
 */
public class FashionMNISTLoaderAndroid(config: FashionMNISTLoaderConfig) : FashionMNISTLoaderCommon(config) {

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
        val client = HttpClient(Android) {
            // No plugins needed for basic functionality
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
         * Creates a new instance of FashionMNISTLoaderAndroid with the default configuration.
         *
         * @return A new instance of FashionMNISTLoaderAndroid.
         */
        public fun create(): FashionMNISTLoaderAndroid {
            return FashionMNISTLoaderAndroid(FashionMNISTLoaderConfig())
        }

        /**
         * Creates a new instance of FashionMNISTLoaderAndroid with a custom cache directory.
         *
         * @param cacheDir The directory to use for caching.
         * @return A new instance of FashionMNISTLoaderAndroid.
         */
        public fun create(cacheDir: String): FashionMNISTLoaderAndroid {
            return FashionMNISTLoaderAndroid(FashionMNISTLoaderConfig(cacheDir = cacheDir))
        }

        /**
         * Creates a new instance of FashionMNISTLoaderAndroid with a custom configuration.
         *
         * @param config The configuration to use.
         * @return A new instance of FashionMNISTLoaderAndroid.
         */
        public fun create(config: FashionMNISTLoaderConfig): FashionMNISTLoaderAndroid {
            return FashionMNISTLoaderAndroid(config)
        }
    }
}
