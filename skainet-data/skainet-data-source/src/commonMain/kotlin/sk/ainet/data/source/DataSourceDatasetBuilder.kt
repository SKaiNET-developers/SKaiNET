package sk.ainet.data.source

/**
 * Kotlin DSL builder for resolving a data source artifact and parsing it into
 * a raw, schema-bearing dataset.
 */
public class DataSourceDatasetBuilder(
    resolver: DataSourceResolver? = null,
    parserRegistry: DataFormatParserRegistry = DataFormatParserRegistry.default()
) {
    private var resolver: DataSourceResolver? = resolver
    private var parserRegistry: DataFormatParserRegistry = parserRegistry
    private var sourceUri: String? = null
    private var requestedFormat: DataFormat? = null
    private var cachePolicy: CachePolicy = CachePolicy.Use
    private var expectedSha256: String? = null
    private val requestHeaders: MutableMap<String, String> = linkedMapOf()
    private var huggingFaceToken: DataSourceAuthToken? = null

    /** Selects the source artifact URI or local path. */
    public fun from(uri: String): DataSourceDatasetBuilder = apply {
        val normalized = uri.trim()
        require(normalized.isNotEmpty()) { "Data source URI must not be blank" }
        sourceUri = normalized
    }

    /** Overrides format inference from the source filename. */
    public fun format(format: DataFormat): DataSourceDatasetBuilder = apply {
        requestedFormat = format
    }

    /** Sets resolver cache behavior for this load. */
    public fun cachePolicy(cachePolicy: CachePolicy): DataSourceDatasetBuilder = apply {
        this.cachePolicy = cachePolicy
    }

    /** Sets an optional expected SHA-256 checksum for resolver verification. */
    public fun expectedSha256(value: String?): DataSourceDatasetBuilder = apply {
        expectedSha256 = value?.trim()?.takeIf { it.isNotEmpty() }
    }

    /** Adds one request header. */
    public fun header(name: String, value: String): DataSourceDatasetBuilder = apply {
        val normalizedName = name.trim()
        require(normalizedName.isNotEmpty()) { "Header name must not be blank" }
        requestHeaders[normalizedName] = value
    }

    /** Adds request headers, replacing entries with the same name. */
    public fun headers(headers: Map<String, String>): DataSourceDatasetBuilder = apply {
        headers.forEach { (name, value) -> header(name, value) }
    }

    /** Sets a provider-specific Hugging Face token for this request. */
    public fun huggingFaceToken(token: DataSourceAuthToken?): DataSourceDatasetBuilder = apply {
        huggingFaceToken = token
    }

    /** Sets a provider-specific Hugging Face token for this request. */
    public fun huggingFaceToken(value: String): DataSourceDatasetBuilder =
        huggingFaceToken(DataSourceAuthToken.from(value))

    /** Supplies the source resolver used to materialize the artifact. */
    public fun resolver(resolver: DataSourceResolver): DataSourceDatasetBuilder = apply {
        this.resolver = resolver
    }

    /** Replaces the parser registry used by this builder. */
    public fun parserRegistry(parserRegistry: DataFormatParserRegistry): DataSourceDatasetBuilder = apply {
        this.parserRegistry = parserRegistry
    }

    /** Registers or replaces one parser in this builder's parser registry. */
    public fun parser(parser: DataFormatParser): DataSourceDatasetBuilder = apply {
        parserRegistry.register(parser)
    }

    /** Resolves and parses the configured source artifact. */
    public suspend fun build(): RawDataset {
        val request = buildRequest()
        val artifact = requireResolver().resolve(request)
        val format = requestedFormat
            ?: DataFormat.inferFromFilename(artifact.filename)
            ?: throw DataSourceException(
                "Cannot infer data format from '${artifact.filename}'. Specify format(...) explicitly."
            )
        val text = artifact.readBytes().decodeToString()
        return parserRegistry.parse(format, text).withSourceMetadata(artifact)
    }

    private fun requireResolver(): DataSourceResolver =
        resolver ?: throw DataSourceException("A DataSourceResolver is required to load a data source dataset")

    private fun buildRequest(): DataSourceRequest {
        val uri = sourceUri ?: throw DataSourceException("Data source URI is required; call from(...) first")
        return DataSourceRequest(
            uri = uri,
            cachePolicy = cachePolicy,
            expectedSha256 = expectedSha256,
            headers = requestHeaders.toMap(),
            huggingFaceToken = huggingFaceToken
        )
    }

    private fun RawDataset.withSourceMetadata(artifact: DataSourceArtifact): RawDataset {
        val sourceMetadata = mutableMapOf(
            "sourceUri" to artifact.request.uri,
            "sourceProvider" to artifact.parsedUri.provider.name,
            "sourceFilename" to artifact.filename,
            "sourceCacheHit" to artifact.cacheHit.toString()
        )
        artifact.localPath?.let { sourceMetadata["sourceLocalPath"] = it }
        artifact.sizeBytes?.let { sourceMetadata["sourceSizeBytes"] = it.toString() }
        return copy(metadata = metadata + sourceMetadata)
    }
}

/** Builds a raw dataset by resolving and parsing a configured data source. */
public suspend fun rawDataset(block: DataSourceDatasetBuilder.() -> Unit): RawDataset =
    DataSourceDatasetBuilder().apply(block).build()

/** Kotlinish alias for [rawDataset]. */
public suspend fun dataset(block: DataSourceDatasetBuilder.() -> Unit): RawDataset =
    rawDataset(block)

/** Builds a raw dataset with this resolver preconfigured. */
public suspend fun DataSourceResolver.rawDataset(block: DataSourceDatasetBuilder.() -> Unit): RawDataset =
    DataSourceDatasetBuilder(this).apply(block).build()

/** Kotlinish alias for [DataSourceResolver.rawDataset]. */
public suspend fun DataSourceResolver.dataset(block: DataSourceDatasetBuilder.() -> Unit): RawDataset =
    rawDataset(block)
