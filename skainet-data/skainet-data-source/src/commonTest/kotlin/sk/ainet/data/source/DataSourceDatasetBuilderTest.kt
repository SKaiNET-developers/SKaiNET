package sk.ainet.data.source

import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class DataSourceDatasetBuilderTest {
    @Test
    fun resolvesSourceAndInfersCsvFormat() = runTest {
        val resolver = FixtureResolver("id,label\n1,cat\n2,dog")
        val token = DataSourceAuthToken.from("hf_token")

        val dataset = resolver.rawDataset {
            from("hf://datasets/org/repo@main/train.csv")
            cachePolicy(CachePolicy.Refresh)
            expectedSha256("sha256")
            header("Accept", "text/csv")
            huggingFaceToken(token)
        }

        assertEquals(listOf("id", "label"), dataset.schema.columns)
        assertEquals(mapOf("id" to "2", "label" to "dog"), dataset.rows[1].values)
        assertEquals("CSV", dataset.metadata["format"])
        assertEquals("hf://datasets/org/repo@main/train.csv", dataset.metadata["sourceUri"])
        assertEquals("HuggingFace", dataset.metadata["sourceProvider"])
        assertEquals("train.csv", dataset.metadata["sourceFilename"])
        assertEquals("false", dataset.metadata["sourceCacheHit"])
        assertEquals("20", dataset.metadata["sourceSizeBytes"])

        val request = resolver.lastRequest
        assertEquals(CachePolicy.Refresh, request?.cachePolicy)
        assertEquals("sha256", request?.expectedSha256)
        assertEquals(mapOf("Accept" to "text/csv"), request?.headers)
        assertEquals(token, request?.huggingFaceToken)
    }

    @Test
    fun explicitFormatOverridesFilenameInference() = runTest {
        val resolver = FixtureResolver("id\tvalue\n1\t42")

        val dataset = resolver.rawDataset {
            from("fixtures/train.txt")
            format(DataFormat.TSV)
        }

        assertEquals(listOf("id", "value"), dataset.schema.columns)
        assertEquals(mapOf("id" to "1", "value" to "42"), dataset.rows.single().values)
    }

    @Test
    fun parserRegistrationReplacesDefaultParser() = runTest {
        val resolver = FixtureResolver("payload")
        val customParser = object : DataFormatParser {
            override val format: DataFormat = DataFormat.CSV

            override fun parse(text: String): RawDataset =
                RawDataset(
                    rows = listOf(RawDataRow(mapOf("raw" to text.uppercase()))),
                    schema = DataSchema(listOf("raw"))
                )
        }

        val dataset = resolver.rawDataset {
            from("fixture.csv")
            parser(customParser)
        }

        assertEquals(mapOf("raw" to "PAYLOAD"), dataset.rows.single().values)
    }

    @Test
    fun failsWhenFormatCannotBeInferred() = runTest {
        val resolver = FixtureResolver("payload")

        assertFailsWith<DataSourceException> {
            resolver.rawDataset {
                from("fixture.bin")
            }
        }
    }

    @Test
    fun failsWhenResolverIsMissing() = runTest {
        assertFailsWith<DataSourceException> {
            rawDataset {
                from("fixture.csv")
            }
        }
    }

    @Test
    fun failsWhenSourceIsMissing() = runTest {
        val resolver = FixtureResolver("payload")

        assertFailsWith<DataSourceException> {
            resolver.rawDataset {
                format(DataFormat.CSV)
            }
        }
    }
}

private class FixtureResolver(
    text: String
) : DataSourceResolver {
    private val bytes = text.encodeToByteArray()

    var lastRequest: DataSourceRequest? = null
        private set

    override suspend fun resolve(request: DataSourceRequest): DataSourceArtifact {
        lastRequest = request
        val parsed = DataSourceUriParser.parse(request.uri)
        return DataSourceArtifact(
            request = request,
            parsedUri = parsed,
            filename = parsed.filename,
            localPath = parsed.localPath,
            sizeBytes = bytes.size.toLong(),
            cacheHit = false,
            sourceOpener = { DataSourceStoredArtifact.inMemory(bytes, parsed.localPath).openSource() }
        )
    }
}
