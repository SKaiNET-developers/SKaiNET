package sk.ainet.data.fashionmnist

import io.ktor.client.HttpClient
import io.ktor.client.engine.js.Js
import io.ktor.client.request.get
import io.ktor.client.statement.HttpResponse
import io.ktor.client.call.body
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.js.ExperimentalWasmJsInterop
import kotlin.js.Promise
import kotlinx.coroutines.await
import sk.ainet.data.common.hasGzipHeader
import sk.ainet.data.common.unsupportedDatasetLoader

@OptIn(ExperimentalWasmJsInterop::class)
@JsFun(
    """
        async function(input) {
            try {
                if (typeof DecompressionStream === 'undefined') return null;
                const ds = new DecompressionStream('gzip');
                const stream = new Blob([input]).stream().pipeThrough(ds);
                const resp = new Response(stream);
                const ab = await resp.arrayBuffer();
                return new Uint8Array(ab);
            } catch (e) {
                return null;
            }
        }
        """
)
private external fun gunzipJs(input: ByteArray): Promise<dynamic>?


/**
 * JS (browser) implementation of the Fashion-MNIST loader.
 */
public class FashionMNISTLoaderJs(config: FashionMNISTLoaderConfig) : FashionMNISTLoaderCommon(config) {

    override suspend fun downloadAndCacheFile(url: String, filename: String): ByteArray =
        withContext(Dispatchers.Default) {
            // No filesystem access in browser JS target; download each time.
            println("[FashionMNIST][JS] Downloading file: $url")
            val gzData = downloadFile(url)

            // Try to gunzip via browser DecompressionStream if available.
            val decompressed = tryGunzip(gzData)
            if (decompressed != null) {
                decompressed
            } else {
                if (gzData.hasGzipHeader()) {
                    unsupportedDatasetLoader(
                        dataset = "Fashion-MNIST",
                        target = "js",
                        reason = "browser DecompressionStream is unavailable; provide an uncompressed IDX URI"
                    )
                }
                gzData
            }
        }

    private suspend fun downloadFile(url: String): ByteArray {
        val client = HttpClient(Js) {}
        try {
            val httpResponse: HttpResponse = client.get(url)
            return httpResponse.body()
        } finally {
            client.close()
        }
    }

    private suspend fun tryGunzip(input: ByteArray): ByteArray? {
        return try {
            gunzipJs(input)?.await()?.unsafeCast<ByteArray?>()
        } catch (t: Throwable) {
            println("[FashionMNIST][JS] Gzip decompression failed: ${t.message}")
            null
        }
    }


    public companion object {
        public fun create(): FashionMNISTLoaderJs = FashionMNISTLoaderJs(FashionMNISTLoaderConfig())
        public fun create(cacheDir: String): FashionMNISTLoaderJs = FashionMNISTLoaderJs(FashionMNISTLoaderConfig(cacheDir = cacheDir))
        public fun create(config: FashionMNISTLoaderConfig): FashionMNISTLoaderJs = FashionMNISTLoaderJs(config)
    }
}
