package sk.ainet.data.source

import kotlinx.coroutines.test.runTest
import kotlinx.io.Buffer
import kotlinx.io.Source
import kotlinx.io.readByteArray
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class DefaultDataSourceResolverTest {
    @Test
    fun resolvesLocalArtifactsThroughStore() = runTest {
        val store = MemoryDataSourceByteStore(
            localArtifacts = mapOf("/data/sample.bin" to "local".encodeToByteArray())
        )
        val fetcher = RecordingFetcher("remote".encodeToByteArray())
        val resolver = DefaultDataSourceResolver(store, fetcher, TestChecksum)

        val artifact = resolver.resolve(DataSourceRequest("/data/sample.bin"))

        assertEquals("/data/sample.bin", artifact.localPath)
        assertTrue(artifact.cacheHit)
        assertEquals(0, fetcher.calls)
        assertContentEquals("local".encodeToByteArray(), artifact.readBytes())

        val copied = Buffer()
        assertEquals(5, artifact.copyTo(copied))
        assertContentEquals("local".encodeToByteArray(), copied.readByteArray())
    }

    @Test
    fun cachesRemoteArtifactsAndReusesThem() = runTest {
        val store = MemoryDataSourceByteStore()
        val fetcher = RecordingFetcher("payload".encodeToByteArray())
        val resolver = DefaultDataSourceResolver(store, fetcher, TestChecksum)
        val request = DataSourceRequest(
            uri = "https://example.test/data.bin",
            expectedSha256 = "sha:payload"
        )

        val first = resolver.resolve(request)
        val second = resolver.resolve(request)

        assertFalse(first.cacheHit)
        assertTrue(second.cacheHit)
        assertEquals(1, fetcher.calls)
        assertEquals(1, store.cacheWrites)
        assertContentEquals("payload".encodeToByteArray(), second.readBytes())
    }

    @Test
    fun bypassSkipsPersistentCache() = runTest {
        val store = MemoryDataSourceByteStore()
        val fetcher = RecordingFetcher("payload".encodeToByteArray())
        val resolver = DefaultDataSourceResolver(store, fetcher, TestChecksum)

        val artifact = resolver.resolve(
            DataSourceRequest(
                uri = "https://example.test/data.bin",
                cachePolicy = CachePolicy.Bypass
            )
        )

        assertEquals(null, artifact.localPath)
        assertFalse(artifact.cacheHit)
        assertEquals(1, fetcher.calls)
        assertEquals(0, store.cacheWrites)
    }

    @Test
    fun verifiesChecksumsInCommonCore() = runTest {
        val store = MemoryDataSourceByteStore()
        val resolver = DefaultDataSourceResolver(
            store = store,
            fetcher = RecordingFetcher("payload".encodeToByteArray()),
            checksum = TestChecksum
        )

        assertFailsWith<DataSourceException> {
            resolver.resolve(
                DataSourceRequest(
                    uri = "https://example.test/data.bin",
                    expectedSha256 = "sha:other"
                )
            )
        }
        assertEquals(0, store.cacheWrites)
    }

    @Test
    fun forwardsProviderHeadersToFetcher() = runTest {
        val fetcher = RecordingFetcher("payload".encodeToByteArray())
        val resolver = DefaultDataSourceResolver(
            store = MemoryDataSourceByteStore(),
            fetcher = fetcher,
            checksum = TestChecksum,
            headerProvider = DataSourceHeaderProvider { request, parsedUri ->
                request.headers + ("X-SKaiNET-Provider" to parsedUri.provider.name)
            }
        )

        resolver.resolve(
            DataSourceRequest(
                uri = "hf://datasets/org/repo@main/file.bin",
                headers = mapOf("Accept" to "application/octet-stream")
            )
        )

        assertEquals(
            mapOf(
                "Accept" to "application/octet-stream",
                "X-SKaiNET-Provider" to "HuggingFace"
            ),
            fetcher.lastHeaders
        )
    }

    @Test
    fun addsHuggingFaceTokenFromRequest() = runTest {
        val fetcher = RecordingFetcher("payload".encodeToByteArray())
        val resolver = DefaultDataSourceResolver(
            store = MemoryDataSourceByteStore(),
            fetcher = fetcher,
            checksum = TestChecksum
        )
        val token = DataSourceAuthToken.from("hf_request")

        resolver.resolve(
            DataSourceRequest(
                uri = "hf://datasets/org/repo@main/file.bin",
                headers = mapOf("Accept" to "application/octet-stream"),
                huggingFaceToken = token
            )
        )

        assertEquals(
            mapOf(
                "Accept" to "application/octet-stream",
                "Authorization" to "Bearer hf_request"
            ),
            fetcher.lastHeaders
        )
        assertEquals("DataSourceAuthToken(***)", token.toString())
    }

    @Test
    fun keepsExistingAuthorizationHeaderOverHuggingFaceToken() = runTest {
        val fetcher = RecordingFetcher("payload".encodeToByteArray())
        val resolver = DefaultDataSourceResolver(
            store = MemoryDataSourceByteStore(),
            fetcher = fetcher,
            checksum = TestChecksum
        )

        resolver.resolve(
            DataSourceRequest(
                uri = "hf://org/repo@main/file.bin",
                headers = mapOf("authorization" to "Bearer explicit"),
                huggingFaceToken = DataSourceAuthToken.from("hf_request")
            )
        )

        assertEquals(mapOf("authorization" to "Bearer explicit"), fetcher.lastHeaders)
    }

    @Test
    fun doesNotAddHuggingFaceTokenToGenericHttp() = runTest {
        val fetcher = RecordingFetcher("payload".encodeToByteArray())
        val resolver = DefaultDataSourceResolver(
            store = MemoryDataSourceByteStore(),
            fetcher = fetcher,
            checksum = TestChecksum
        )

        resolver.resolve(
            DataSourceRequest(
                uri = "https://example.test/data.bin",
                huggingFaceToken = DataSourceAuthToken.from("hf_request")
            )
        )

        assertEquals(emptyMap(), fetcher.lastHeaders)
    }
}

private class MemoryDataSourceByteStore(
    private val localArtifacts: Map<String, ByteArray> = emptyMap()
) : DataSourceArtifactStore {
    private val cacheArtifacts = mutableMapOf<String, DataSourceStoredArtifact>()

    var cacheWrites: Int = 0
        private set

    override suspend fun readLocal(path: String): DataSourceStoredArtifact? {
        return localArtifacts[path]?.storedAt(path)
    }

    override suspend fun readCache(cacheKey: String): DataSourceStoredArtifact? {
        return cacheArtifacts[cacheKey]
    }

    override suspend fun writeCache(
        cacheKey: String,
        source: Source,
        sizeBytes: Long?,
        validate: suspend (DataSourceStoredArtifact) -> Unit
    ): DataSourceStoredArtifact {
        val stored = DataSourceStoredArtifact.inMemoryFrom(
            source = source,
            localPath = "/cache/$cacheKey",
            sizeBytes = sizeBytes
        )
        validate(stored)
        cacheWrites++
        cacheArtifacts[cacheKey] = stored
        return stored
    }

    private fun ByteArray.storedAt(path: String): DataSourceStoredArtifact {
        val bytes = copyOf()
        return DataSourceStoredArtifact.inMemory(bytes, localPath = path)
    }
}

private class RecordingFetcher(
    private val bytes: ByteArray
) : RemoteDataSourceFetcher {
    var calls: Int = 0
        private set

    var lastHeaders: Map<String, String> = emptyMap()
        private set

    override suspend fun fetch(uri: String, headers: Map<String, String>): DataSourceRemoteContent {
        calls++
        lastHeaders = headers
        return DataSourceRemoteContent.fromBytes(bytes.copyOf())
    }
}

private object TestChecksum : DataSourceChecksum {
    override fun newSha256(): DataSourceHash = TestHash()
}

private class TestHash : DataSourceHash {
    private val text = StringBuilder()

    override fun update(bytes: ByteArray, startIndex: Int, endIndex: Int) {
        text.append(bytes.copyOfRange(startIndex, endIndex).decodeToString())
    }

    override fun hex(): String = "sha:$text"
}
