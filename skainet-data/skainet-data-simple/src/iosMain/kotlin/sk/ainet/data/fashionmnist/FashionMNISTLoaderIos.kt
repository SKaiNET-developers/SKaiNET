package sk.ainet.data.fashionmnist

import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.engine.darwin.Darwin
import io.ktor.client.request.get
import io.ktor.client.statement.HttpResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import sk.ainet.data.common.hasGzipHeader
import sk.ainet.data.common.unsupportedDatasetLoader

/**
 * iOS implementation of the Fashion-MNIST loader.
 *
 * @property config The configuration for the Fashion-MNIST loader.
 */
public class FashionMNISTLoaderIos(config: FashionMNISTLoaderConfig) : FashionMNISTLoaderCommon(config) {

    /**
     * Downloads and caches a file.
     *
     * @param url The URL to download from.
     * @param filename The name of the file to save.
     * @return The bytes of the decompressed file.
     */
    override suspend fun downloadAndCacheFile(url: String, filename: String): ByteArray =
        withContext(Dispatchers.Default) {
            // In this simplified iOS implementation, we don't cache files
            // We'll just download the file every time
            println("Downloading Fashion-MNIST file: $url")
            val data = downloadFile(url)

            if (data.hasGzipHeader()) {
                unsupportedDatasetLoader(
                    dataset = "Fashion-MNIST",
                    target = "ios",
                    reason = "gzip decompression is not implemented; provide an uncompressed IDX URI"
                )
            }

            return@withContext data
        }

    /**
     * Downloads a file from a URL.
     *
     * @param url The URL to download from.
     * @return The bytes of the file.
     */
    private suspend fun downloadFile(url: String): ByteArray {
        val client = HttpClient(Darwin) {
            // No plugins needed for basic functionality
        }

        try {
            val httpResponse: HttpResponse = client.get(url)
            return httpResponse.body()
        } finally {
            client.close()
        }
    }

    public companion object {
        /**
         * Creates a new instance of FashionMNISTLoaderIos with the default configuration.
         *
         * @return A new instance of FashionMNISTLoaderIos.
         */
        public fun create(): FashionMNISTLoaderIos {
            return FashionMNISTLoaderIos(FashionMNISTLoaderConfig())
        }

        /**
         * Creates a new instance of FashionMNISTLoaderIos with a custom cache directory.
         *
         * @param cacheDir The directory to use for caching.
         * @return A new instance of FashionMNISTLoaderIos.
         */
        public fun create(cacheDir: String): FashionMNISTLoaderIos {
            return FashionMNISTLoaderIos(FashionMNISTLoaderConfig(cacheDir = cacheDir))
        }

        /**
         * Creates a new instance of FashionMNISTLoaderIos with a custom configuration.
         *
         * @param config The configuration to use.
         * @return A new instance of FashionMNISTLoaderIos.
         */
        public fun create(config: FashionMNISTLoaderConfig): FashionMNISTLoaderIos {
            return FashionMNISTLoaderIos(config)
        }
    }
}
