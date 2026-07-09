package sk.ainet.data.cifar10

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import sk.ainet.data.common.JvmDatasetSourceReader
import sk.ainet.data.common.gunzip
import java.io.File
import java.io.FileOutputStream

/**
 * JVM implementation of the CIFAR-10 loader.
 *
 * Downloads the CIFAR-10 binary archive, extracts batch files, and caches them locally.
 *
 * @property config The configuration for the CIFAR-10 loader.
 */
public class CIFAR10LoaderJvm(config: CIFAR10LoaderConfig) : CIFAR10LoaderCommon(config) {
    private val sources = JvmDatasetSourceReader(
        cacheDir = config.cacheDir,
        useCache = config.useCache,
        huggingFaceTokenProvider = config.huggingFaceTokenProvider,
        useEnvironmentHuggingFaceToken = config.useEnvironmentHuggingFaceToken
    )

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

        // Check if we need to resolve and extract the archive
        if (!extractedDir.exists() || !config.useCache) {
            println("Extracting CIFAR-10 archive...")
            extractTarGz(sources.read(config.archiveUri), cacheDir.path)
        }

        if (!batchFile.exists()) {
            throw IllegalStateException("Batch file not found after extraction: ${batchFile.path}")
        }

        return@withContext batchFile.readBytes()
    }

    /**
     * Extracts a .tar.gz archive using a simple TAR parser.
     *
     * @param archiveBytes The bytes of the .tar.gz file.
     * @param outputDir The directory to extract files to.
     */
    private fun extractTarGz(archiveBytes: ByteArray, outputDir: String) {
        val outputDirFile = File(outputDir)
        extractTar(archiveBytes.gunzip(), outputDirFile)
    }

    /**
     * Extracts files from a TAR archive.
     *
     * TAR format:
     * - 512-byte header blocks
     * - File content (rounded up to 512-byte blocks)
     * - Two empty 512-byte blocks at the end
     */
    private fun extractTar(tarBytes: ByteArray, outputDir: File) {
        var offset = 0
        val headerSize = 512

        while (offset + headerSize <= tarBytes.size) {
            // Check for end of archive (two zero blocks)
            if (isZeroBlock(tarBytes, offset, headerSize)) {
                break
            }

            // Parse header
            val filename = parseString(tarBytes, offset, 100).trimEnd('\u0000', '/')
            val fileSizeOctal = parseString(tarBytes, offset + 124, 12).trim('\u0000', ' ')
            val typeFlag = tarBytes[offset + 156].toInt().toChar()

            val fileSize = if (fileSizeOctal.isNotEmpty()) {
                fileSizeOctal.toLongOrNull(8) ?: 0L
            } else {
                0L
            }

            // Move past header
            offset += headerSize

            // Skip if it's a directory or special file
            if (typeFlag == '5' || typeFlag == 'x' || typeFlag == 'g' || filename.isEmpty()) {
                // Round up to next 512-byte boundary
                offset += ((fileSize + 511) / 512 * 512).toInt()
                continue
            }

            // Only extract regular files (type '0' or '\0')
            if (typeFlag == '0' || typeFlag == '\u0000') {
                val outputFile = File(outputDir, filename)

                // Ensure parent directory exists
                outputFile.parentFile?.mkdirs()

                // Extract file content
                if (fileSize > 0 && offset + fileSize <= tarBytes.size) {
                    FileOutputStream(outputFile).use { fos ->
                        fos.write(tarBytes, offset, fileSize.toInt())
                    }
                    println("Extracted: ${outputFile.path}")
                }
            }

            // Move to next header (content is padded to 512-byte boundary)
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
        /**
         * Creates a new instance of CIFAR10LoaderJvm with the default configuration.
         *
         * @return A new instance of CIFAR10LoaderJvm.
         */
        public fun create(): CIFAR10LoaderJvm {
            return CIFAR10LoaderJvm(CIFAR10LoaderConfig())
        }

        /**
         * Creates a new instance of CIFAR10LoaderJvm with a custom cache directory.
         *
         * @param cacheDir The directory to use for caching.
         * @return A new instance of CIFAR10LoaderJvm.
         */
        public fun create(cacheDir: String): CIFAR10LoaderJvm {
            return CIFAR10LoaderJvm(CIFAR10LoaderConfig(cacheDir = cacheDir))
        }

        /**
         * Creates a new instance of CIFAR10LoaderJvm with a custom configuration.
         *
         * @param config The configuration to use.
         * @return A new instance of CIFAR10LoaderJvm.
         */
        public fun create(config: CIFAR10LoaderConfig): CIFAR10LoaderJvm {
            return CIFAR10LoaderJvm(config)
        }
    }
}
