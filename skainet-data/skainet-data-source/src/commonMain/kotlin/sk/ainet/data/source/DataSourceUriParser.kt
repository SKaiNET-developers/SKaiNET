package sk.ainet.data.source

/**
 * Parses SKaiNET data source URIs.
 *
 * Supported forms:
 * - `file:///absolute/path`
 * - `/absolute/or/relative/path`
 * - `https://host/path`
 * - `hf+https://huggingface.co/org/repo/resolve/main/file`
 * - `hf://org/repo@revision/path/to/file`
 * - `hf://datasets/org/repo@revision/path/to/file`
 */
public object DataSourceUriParser {
    public fun parse(uri: String): ParsedDataSourceUri {
        val raw = uri.trim()
        require(raw.isNotEmpty()) { "Data source URI must not be blank" }

        return when {
            raw.startsWith(HF_HTTPS_PREFIX) -> parseHfHttps(raw)
            raw.startsWith(HF_URI_PREFIX) -> parseHfUri(raw)
            raw.startsWith(FILE_URI_PREFIX) -> parseFileUri(raw)
            raw.startsWith(HTTPS_PREFIX) || raw.startsWith(HTTP_PREFIX) -> parseHttp(raw)
            raw.contains("://") -> throw UnsupportedDataSourceUriException(raw)
            else -> parsePlainFilePath(raw)
        }
    }

    private fun parseFileUri(raw: String): ParsedDataSourceUri {
        val localPath = normalizeFileUriPath(raw.removePrefix(FILE_URI_PREFIX))
        val filename = extractFilename(localPath)
        return ParsedDataSourceUri(
            rawUri = raw,
            provider = DataSourceProvider.File,
            transportUri = raw,
            filename = filename,
            cacheKey = cacheKey(DataSourceProvider.File, localPath, filename),
            localPath = localPath
        )
    }

    private fun parsePlainFilePath(raw: String): ParsedDataSourceUri {
        val filename = extractFilename(raw)
        return ParsedDataSourceUri(
            rawUri = raw,
            provider = DataSourceProvider.File,
            transportUri = raw,
            filename = filename,
            cacheKey = cacheKey(DataSourceProvider.File, raw, filename),
            localPath = raw
        )
    }

    private fun parseHttp(raw: String): ParsedDataSourceUri {
        val filename = extractFilename(raw)
        return ParsedDataSourceUri(
            rawUri = raw,
            provider = DataSourceProvider.Http,
            transportUri = raw,
            filename = filename,
            cacheKey = cacheKey(DataSourceProvider.Http, raw, filename)
        )
    }

    private fun parseHfHttps(raw: String): ParsedDataSourceUri {
        val transportUri = raw.removePrefix("hf+")
        val filename = extractFilename(transportUri)
        return ParsedDataSourceUri(
            rawUri = raw,
            provider = DataSourceProvider.HuggingFace,
            transportUri = transportUri,
            filename = filename,
            cacheKey = cacheKey(DataSourceProvider.HuggingFace, transportUri, filename),
            huggingFace = HuggingFaceLocation(
                repoType = HuggingFaceRepoType.Model,
                repoId = null,
                revision = null,
                path = null
            )
        )
    }

    private fun parseHfUri(raw: String): ParsedDataSourceUri {
        val body = raw.removePrefix(HF_URI_PREFIX).trim('/')
        val segments = body.split('/').filter { it.isNotBlank() }
        require(segments.size >= 3) {
            "hf:// URI must include repo owner, repo name, and file path: $raw"
        }

        val (repoType, repoStart) = when (segments.first()) {
            "models", "model" -> HuggingFaceRepoType.Model to 1
            "datasets", "dataset" -> HuggingFaceRepoType.Dataset to 1
            "spaces", "space" -> HuggingFaceRepoType.Space to 1
            else -> HuggingFaceRepoType.Model to 0
        }
        require(segments.size - repoStart >= 3) {
            "hf:// URI must include repo owner, repo name, and file path: $raw"
        }

        val owner = segments[repoStart]
        val repoAndRevision = segments[repoStart + 1]
        val repoName = repoAndRevision.substringBefore('@')
        val revision = repoAndRevision.substringAfter('@', "main")
        val filePath = segments.drop(repoStart + 2).joinToString("/")
        val repoId = "$owner/$repoName"
        val prefix = when (repoType) {
            HuggingFaceRepoType.Model -> ""
            HuggingFaceRepoType.Dataset -> "datasets/"
            HuggingFaceRepoType.Space -> "spaces/"
        }
        val transportUri = "https://huggingface.co/$prefix$repoId/resolve/$revision/$filePath"
        val filename = extractFilename(filePath)

        return ParsedDataSourceUri(
            rawUri = raw,
            provider = DataSourceProvider.HuggingFace,
            transportUri = transportUri,
            filename = filename,
            cacheKey = cacheKey(DataSourceProvider.HuggingFace, transportUri, filename),
            huggingFace = HuggingFaceLocation(
                repoType = repoType,
                repoId = repoId,
                revision = revision,
                path = filePath
            )
        )
    }

    private fun normalizeFileUriPath(path: String): String {
        val withoutLocalhost = path.removePrefix("localhost/")
        val normalized = if (withoutLocalhost.startsWith("/")) withoutLocalhost else "/$withoutLocalhost"
        return percentDecode(normalized)
    }

    private fun extractFilename(value: String): String {
        val withoutFragment = value.substringBefore('#').substringBefore('?').trimEnd('/')
        val filename = withoutFragment.substringAfterLast('/', missingDelimiterValue = withoutFragment)
        return percentDecode(filename).ifBlank { "artifact" }
    }

    private fun percentDecode(value: String): String {
        val out = StringBuilder(value.length)
        var i = 0
        while (i < value.length) {
            val c = value[i]
            if (c == '%' && i + 2 < value.length) {
                val decoded = hexByte(value[i + 1], value[i + 2])
                if (decoded != null) {
                    out.append(decoded.toInt().toChar())
                    i += 3
                    continue
                }
            }
            out.append(c)
            i++
        }
        return out.toString()
    }

    private fun hexByte(high: Char, low: Char): Byte? {
        val hi = high.digitToIntOrNull(16) ?: return null
        val lo = low.digitToIntOrNull(16) ?: return null
        return ((hi shl 4) or lo).toByte()
    }

    private fun cacheKey(provider: DataSourceProvider, normalizedUri: String, filename: String): String {
        val safeName = filename.map { ch ->
            if (ch.isLetterOrDigit() || ch == '.' || ch == '-' || ch == '_') ch else '_'
        }.joinToString("")
        return "${provider.name.lowercase()}-${fnv1a32Hex(normalizedUri)}-$safeName"
    }

    private fun fnv1a32Hex(value: String): String {
        var hash = FNV_OFFSET
        val bytes = value.encodeToByteArray()
        for (byte in bytes) {
            hash = hash xor (byte.toInt() and 0xff)
            hash *= FNV_PRIME
        }
        return hash.toHex8()
    }

    private fun Int.toHex8(): String {
        val chars = CharArray(8)
        for (i in chars.indices) {
            val shift = (7 - i) * 4
            chars[i] = HEX[(this ushr shift) and 0x0f]
        }
        return chars.concatToString()
    }

    private const val FILE_URI_PREFIX = "file://"
    private const val HTTP_PREFIX = "http://"
    private const val HTTPS_PREFIX = "https://"
    private const val HF_HTTPS_PREFIX = "hf+https://"
    private const val HF_URI_PREFIX = "hf://"
    private const val FNV_OFFSET = -2128831035
    private const val FNV_PRIME = 16777619
    private val HEX: CharArray = "0123456789abcdef".toCharArray()
}
