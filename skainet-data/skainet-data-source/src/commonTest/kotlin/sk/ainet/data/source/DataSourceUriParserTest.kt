package sk.ainet.data.source

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertNull

class DataSourceUriParserTest {
    @Test
    fun parsesFileUri() {
        val parsed = DataSourceUriParser.parse("file:///tmp/skainet/train-images.idx")

        assertEquals(DataSourceProvider.File, parsed.provider)
        assertEquals("/tmp/skainet/train-images.idx", parsed.localPath)
        assertEquals("train-images.idx", parsed.filename)
    }

    @Test
    fun parsesPlainPathAsFile() {
        val parsed = DataSourceUriParser.parse("fixtures/mnist/train-labels.idx")

        assertEquals(DataSourceProvider.File, parsed.provider)
        assertEquals("fixtures/mnist/train-labels.idx", parsed.localPath)
        assertEquals("train-labels.idx", parsed.filename)
    }

    @Test
    fun parsesHttpUri() {
        val parsed = DataSourceUriParser.parse("https://example.test/data/sample.csv?download=1")

        assertEquals(DataSourceProvider.Http, parsed.provider)
        assertEquals("sample.csv", parsed.filename)
        assertNull(parsed.huggingFace)
    }

    @Test
    fun parsesHuggingFaceHttpsProviderPrefix() {
        val parsed = DataSourceUriParser.parse(
            "hf+https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json"
        )

        assertEquals(DataSourceProvider.HuggingFace, parsed.provider)
        assertEquals(
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json",
            parsed.transportUri
        )
        assertEquals("tokenizer.json", parsed.filename)
    }

    @Test
    fun parsesHuggingFaceDatasetShorthand() {
        val parsed = DataSourceUriParser.parse("hf://datasets/mnist/mnist@main/plain_text/train-00000.parquet")

        assertEquals(DataSourceProvider.HuggingFace, parsed.provider)
        assertEquals(HuggingFaceRepoType.Dataset, parsed.huggingFace?.repoType)
        assertEquals("mnist/mnist", parsed.huggingFace?.repoId)
        assertEquals("main", parsed.huggingFace?.revision)
        assertEquals("plain_text/train-00000.parquet", parsed.huggingFace?.path)
        assertEquals(
            "https://huggingface.co/datasets/mnist/mnist/resolve/main/plain_text/train-00000.parquet",
            parsed.transportUri
        )
    }

    @Test
    fun cacheKeyDependsOnNormalizedUri() {
        val first = DataSourceUriParser.parse("https://example.test/a.txt")
        val second = DataSourceUriParser.parse("https://example.test/b.txt")

        assertNotEquals(first.cacheKey, second.cacheKey)
    }

    @Test
    fun rejectsUnknownSchemes() {
        assertFailsWith<UnsupportedDataSourceUriException> {
            DataSourceUriParser.parse("s3://bucket/object")
        }
    }
}
