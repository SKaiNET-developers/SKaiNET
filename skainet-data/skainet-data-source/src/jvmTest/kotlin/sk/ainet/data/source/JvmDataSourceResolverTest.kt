package sk.ainet.data.source

import kotlinx.coroutines.test.runTest
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class JvmDataSourceResolverTest {
    @Test
    fun resolvesLocalFileUri() = runTest {
        val root = Files.createTempDirectory("skainet-data-source-test").toFile()
        try {
            val file = root.resolve("sample.txt")
            file.writeText("hello")
            val resolver = JvmDataSourceResolver(cacheDir = root.resolve("cache"))

            val artifact = resolver.resolve(DataSourceRequest(file.toURI().toString()))

            assertEquals("sample.txt", artifact.filename)
            assertEquals(file.canonicalPath, artifact.localPath)
            assertTrue(artifact.cacheHit)
            assertContentEquals("hello".encodeToByteArray(), artifact.readBytes())
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun cachesRemoteArtifacts() = runTest {
        val root = Files.createTempDirectory("skainet-data-source-test").toFile()
        try {
            val fetcher = FakeFetcher("first".encodeToByteArray())
            val resolver = JvmDataSourceResolver(cacheDir = root.resolve("cache"), fetcher = fetcher)
            val request = DataSourceRequest(
                "hf+https://huggingface.co/example/model/resolve/main/config.json"
            )

            val first = resolver.resolve(request)
            val second = resolver.resolve(request)

            assertEquals(1, fetcher.calls)
            assertFalse(first.cacheHit)
            assertTrue(second.cacheHit)
            assertNotNull(second.localPath)
            assertContentEquals("first".encodeToByteArray(), second.readBytes())
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun refreshFetchesAgain() = runTest {
        val root = Files.createTempDirectory("skainet-data-source-test").toFile()
        try {
            val fetcher = QueueFetcher(
                "old".encodeToByteArray(),
                "new".encodeToByteArray()
            )
            val resolver = JvmDataSourceResolver(cacheDir = root.resolve("cache"), fetcher = fetcher)
            val uri = "https://example.test/data.bin"

            resolver.resolve(DataSourceRequest(uri)).readBytes()
            val refreshed = resolver.resolve(DataSourceRequest(uri, cachePolicy = CachePolicy.Refresh))

            assertEquals(2, fetcher.calls)
            assertContentEquals("new".encodeToByteArray(), refreshed.readBytes())
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun offlineFailsWhenCacheIsMissing() = runTest {
        val root = Files.createTempDirectory("skainet-data-source-test").toFile()
        try {
            val resolver = JvmDataSourceResolver(cacheDir = root.resolve("cache"), fetcher = FakeFetcher(ByteArray(0)))

            assertFailsWith<DataSourceException> {
                resolver.resolve(
                    DataSourceRequest(
                        uri = "https://example.test/missing.bin",
                        cachePolicy = CachePolicy.Offline
                    )
                )
            }
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun bypassDoesNotWriteCache() = runTest {
        val root = Files.createTempDirectory("skainet-data-source-test").toFile()
        try {
            val fetcher = FakeFetcher("bytes".encodeToByteArray())
            val cacheDir = root.resolve("cache")
            val resolver = JvmDataSourceResolver(cacheDir = cacheDir, fetcher = fetcher)

            val artifact = resolver.resolve(
                DataSourceRequest("https://example.test/data.bin", cachePolicy = CachePolicy.Bypass)
            )

            assertEquals(1, fetcher.calls)
            assertEquals(null, artifact.localPath)
            assertFalse(cacheDir.exists())
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun verifiesSha256() = runTest {
        val root = Files.createTempDirectory("skainet-data-source-test").toFile()
        try {
            val resolver = JvmDataSourceResolver(
                cacheDir = root.resolve("cache"),
                fetcher = FakeFetcher("payload".encodeToByteArray())
            )

            assertFailsWith<DataSourceException> {
                resolver.resolve(
                    DataSourceRequest(
                        uri = "https://example.test/payload.bin",
                        expectedSha256 = "0000"
                    )
                )
            }
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun sendsConfiguredHuggingFaceToken() = runTest {
        val root = Files.createTempDirectory("skainet-data-source-test").toFile()
        try {
            val fetcher = FakeFetcher("payload".encodeToByteArray())
            val resolver = JvmDataSourceResolver(
                cacheDir = root.resolve("cache"),
                fetcher = fetcher,
                huggingFaceToken = DataSourceAuthToken.from("hf_configured"),
                useEnvironmentHuggingFaceToken = false
            )

            resolver.resolve(DataSourceRequest("hf://org/repo@main/file.bin"))

            assertEquals(mapOf("Authorization" to "Bearer hf_configured"), fetcher.lastHeaders)
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun requestHuggingFaceTokenOverridesConfiguredToken() = runTest {
        val root = Files.createTempDirectory("skainet-data-source-test").toFile()
        try {
            val fetcher = FakeFetcher("payload".encodeToByteArray())
            val resolver = JvmDataSourceResolver(
                cacheDir = root.resolve("cache"),
                fetcher = fetcher,
                huggingFaceToken = DataSourceAuthToken.from("hf_configured"),
                useEnvironmentHuggingFaceToken = false
            )

            resolver.resolve(
                DataSourceRequest(
                    uri = "hf://org/repo@main/file.bin",
                    huggingFaceToken = DataSourceAuthToken.from("hf_request")
                )
            )

            assertEquals(mapOf("Authorization" to "Bearer hf_request"), fetcher.lastHeaders)
        } finally {
            root.deleteRecursively()
        }
    }
}

private class FakeFetcher(
    private val bytes: ByteArray
) : RemoteDataSourceFetcher {
    var calls: Int = 0
        private set

    var lastHeaders: Map<String, String> = emptyMap()
        private set

    override suspend fun fetch(uri: String, headers: Map<String, String>): DataSourceRemoteContent {
        calls++
        lastHeaders = headers
        return DataSourceRemoteContent.fromBytes(bytes)
    }
}

private class QueueFetcher(
    private vararg val responses: ByteArray
) : RemoteDataSourceFetcher {
    var calls: Int = 0
        private set

    override suspend fun fetch(uri: String, headers: Map<String, String>): DataSourceRemoteContent {
        val index = calls.coerceAtMost(responses.lastIndex)
        calls++
        return DataSourceRemoteContent.fromBytes(responses[index])
    }
}
