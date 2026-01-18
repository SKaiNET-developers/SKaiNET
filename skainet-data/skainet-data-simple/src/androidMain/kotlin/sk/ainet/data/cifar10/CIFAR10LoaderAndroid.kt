package sk.ainet.data.cifar10

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
 * Android implementation of the CIFAR-10 loader.
 *
 * Downloads the CIFAR-10 binary archive, extracts batch files, and caches them locally.
 *
 * @property config The configuration for the CIFAR-10 loader.
 */
public class CIFAR10LoaderAndroid(config: CIFAR10LoaderConfig) : CIFAR10LoaderCommon(config) {

    /**
     * Downloads the CIFAR-10 archive and extracts the specified batch file.
     *
     * @param batchFilename The name of the batch file to extract.
     * @return The bytes of the extracted batch file.
     */
    override suspend fun downloadAndExtractBatch(batchFilename: String): ByteArray = withContext(Dispatchers.IO) {
        val cacheDir = File(config.cacheDir)
        if (!cacheDir.exists()) {
            cacheDir.mkdirs()
        }

        val extractedDir = File(cacheDir, "cifar-10-batches-bin")
        val batchFile = File(extractedDir, batchFilename)

        // Check if the batch file already exists in cache
        if (config.useCache && batchFile.exists()) {
            println("Using cached file: ${batchFile.path}")
            return@withContext batchFile.readBytes()
        }

        // Check if we need to download and extract the archive
        if (!extractedDir.exists() || !config.useCache) {
            val archiveFile = File(cacheDir, CIFAR10Constants.ARCHIVE_FILENAME)

            // Download if not cached
            if (!archiveFile.exists() || !config.useCache) {
                println("Downloading CIFAR-10 archive: ${CIFAR10Constants.DOWNLOAD_URL}")
                downloadFile(CIFAR10Constants.DOWNLOAD_URL, archiveFile.path)
            } else {
                println("Using cached archive: ${archiveFile.path}")
            }

            // Extract the archive
            println("Extracting CIFAR-10 archive...")
            extractTarGz(archiveFile.path, cacheDir.path)
        }

        if (!batchFile.exists()) {
            throw IllegalStateException("Batch file not found after extraction: ${batchFile.path}")
        }

        return@withContext batchFile.readBytes()
    }

    private suspend fun downloadFile(url: String, outputPath: String) {
        val client = HttpClient(Android) {
            // No plugins needed for basic functionality
        }

        try {
            val file = File(outputPath)

            val httpResponse: HttpResponse = client.get(url)
            val responseBody: ByteArray = httpResponse.body()
            file.writeBytes(responseBody)

            println("File saved to ${file.path} (${responseBody.size} bytes)")
        } finally {
            client.close()
        }
    }

    private fun extractTarGz(archivePath: String, outputDir: String) {
        val outputDirFile = File(outputDir)

        val tarBytes = GZIPInputStream(FileInputStream(archivePath)).use { gzipIn ->
            gzipIn.readBytes()
        }

        extractTar(tarBytes, outputDirFile)
    }

    private fun extractTar(tarBytes: ByteArray, outputDir: File) {
        var offset = 0
        val headerSize = 512

        while (offset + headerSize <= tarBytes.size) {
            if (isZeroBlock(tarBytes, offset, headerSize)) {
                break
            }

            val filename = parseString(tarBytes, offset, 100).trimEnd('\u0000', '/')
            val fileSizeOctal = parseString(tarBytes, offset + 124, 12).trim('\u0000', ' ')
            val typeFlag = tarBytes[offset + 156].toInt().toChar()

            val fileSize = if (fileSizeOctal.isNotEmpty()) {
                fileSizeOctal.toLongOrNull(8) ?: 0L
            } else {
                0L
            }

            offset += headerSize

            if (typeFlag == '5' || typeFlag == 'x' || typeFlag == 'g' || filename.isEmpty()) {
                offset += ((fileSize + 511) / 512 * 512).toInt()
                continue
            }

            if (typeFlag == '0' || typeFlag == '\u0000') {
                val outputFile = File(outputDir, filename)
                outputFile.parentFile?.mkdirs()

                if (fileSize > 0 && offset + fileSize <= tarBytes.size) {
                    FileOutputStream(outputFile).use { fos ->
                        fos.write(tarBytes, offset, fileSize.toInt())
                    }
                    println("Extracted: ${outputFile.path}")
                }
            }

            offset += ((fileSize + 511) / 512 * 512).toInt()
        }
    }

    private fun isZeroBlock(bytes: ByteArray, offset: Int, size: Int): Boolean {
        for (i in 0 until size) {
            if (offset + i >= bytes.size) return true
            if (bytes[offset + i] != 0.toByte()) return false
        }
        return true
    }

    private fun parseString(bytes: ByteArray, offset: Int, length: Int): String {
        val sb = StringBuilder()
        for (i in 0 until length) {
            if (offset + i >= bytes.size) break
            val b = bytes[offset + i]
            if (b == 0.toByte()) break
            sb.append(b.toInt().toChar())
        }
        return sb.toString()
    }

    public companion object {
        public fun create(): CIFAR10LoaderAndroid = CIFAR10LoaderAndroid(CIFAR10LoaderConfig())
        public fun create(cacheDir: String): CIFAR10LoaderAndroid = CIFAR10LoaderAndroid(CIFAR10LoaderConfig(cacheDir = cacheDir))
        public fun create(config: CIFAR10LoaderConfig): CIFAR10LoaderAndroid = CIFAR10LoaderAndroid(config)
    }
}
