package sk.ainet.data.source

import kotlinx.coroutines.test.runTest
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
        val resolver = DefaultDataSourceResolver(
            store = MemoryDataSourceByteStore(),
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
}

private class MemoryDataSourceByteStore(
    private val localArtifacts: Map<String, ByteArray> = emptyMap()
) : DataSourceByteStore {
    private val cacheArtifacts = mutableMapOf<String, ByteArray>()

    var cacheWrites: Int = 0
        private set

    override suspend fun readLocal(path: String): DataSourceStoredArtifact? {
        return localArtifacts[path]?.storedAt(path)
    }

    override suspend fun readCache(cacheKey: String): DataSourceStoredArtifact? {
        return cacheArtifacts[cacheKey]?.storedAt("/cache/$cacheKey")
    }

    override suspend fun writeCache(cacheKey: String, bytes: ByteArray): DataSourceStoredArtifact {
        cacheWrites++
        cacheArtifacts[cacheKey] = bytes
        return bytes.storedAt("/cache/$cacheKey")
    }

    private fun ByteArray.storedAt(path: String): DataSourceStoredArtifact {
        val bytes = copyOf()
        return DataSourceStoredArtifact(
            localPath = path,
            sizeBytes = bytes.size.toLong(),
            byteReader = { bytes.copyOf() }
        )
    }
}

private class RecordingFetcher(
    private val bytes: ByteArray
) : RemoteDataSourceFetcher {
    var calls: Int = 0
        private set

    var lastHeaders: Map<String, String> = emptyMap()
        private set

    override suspend fun fetch(uri: String, headers: Map<String, String>): ByteArray {
        calls++
        lastHeaders = headers
        return bytes.copyOf()
    }
}

private object TestChecksum : DataSourceChecksum {
    override fun sha256Hex(bytes: ByteArray): String = "sha:${bytes.decodeToString()}"
}
