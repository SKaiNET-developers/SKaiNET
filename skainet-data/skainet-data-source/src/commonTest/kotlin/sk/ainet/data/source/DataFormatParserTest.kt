package sk.ainet.data.source

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame

class DataFormatParserTest {
    @Test
    fun parsesCsvHeaderAndQuotedValues() {
        val dataset = DataFormatParserRegistry.default().parse(
            DataFormat.CSV,
            "name,label,comment\n" +
                "Ada,math,\"first, second\"\n" +
                "Grace,compiler,\"said \"\"hello\"\"\""
        )

        assertEquals(listOf("name", "label", "comment"), dataset.schema.columns)
        assertEquals(2, dataset.rows.size)
        assertEquals("Ada", dataset.rows[0].values["name"])
        assertEquals("first, second", dataset.rows[0].values["comment"])
        assertEquals("said \"hello\"", dataset.rows[1].values["comment"])
        assertEquals("CSV", dataset.metadata["format"])
        assertEquals("2", dataset.metadata["rowCount"])
    }

    @Test
    fun parsesTsvRows() {
        val dataset = DataFormatParserRegistry.default().parse(
            DataFormat.TSV,
            "id\tvalue\n" +
                "1\t42\n" +
                "2\t84"
        )

        assertEquals(listOf("id", "value"), dataset.schema.columns)
        assertEquals(
            mapOf("id" to "2", "value" to "84"),
            dataset.rows[1].values
        )
    }

    @Test
    fun parsesJsonLinesWithUnionSchema() {
        val dataset = DataFormatParserRegistry.default().parse(
            DataFormat.JSON_LINES,
            "{\"id\":1,\"label\":\"cat\",\"pixels\":[0,1],\"meta\":{\"split\":\"train\"}}\n" +
                "{\"id\":2,\"label\":\"dog\",\"score\":0.5}"
        )

        assertEquals(listOf("id", "label", "pixels", "meta", "score"), dataset.schema.columns)
        assertEquals("1", dataset.rows[0].values["id"])
        assertEquals("[0,1]", dataset.rows[0].values["pixels"])
        assertEquals("{\"split\":\"train\"}", dataset.rows[0].values["meta"])
        assertEquals("", dataset.rows[0].values["score"])
        assertEquals("0.5", dataset.rows[1].values["score"])
    }

    @Test
    fun replacesRegisteredParser() {
        val registry = DataFormatParserRegistry(parsers = emptyList())
        val parser = object : DataFormatParser {
            override val format: DataFormat = DataFormat.CSV

            override fun parse(text: String): RawDataset =
                RawDataset(
                    rows = listOf(RawDataRow(mapOf("value" to text))),
                    schema = DataSchema(listOf("value"))
                )
        }

        registry.register(parser)

        assertSame(parser, registry.parserFor(DataFormat.CSV))
        assertEquals("payload", registry.parse(DataFormat.CSV, "payload").rows.single().values["value"])
        assertFailsWith<DataSourceException> {
            registry.parserFor(DataFormat.TSV)
        }
    }

    @Test
    fun rejectsDelimitedRowsWithWrongWidth() {
        assertFailsWith<IllegalArgumentException> {
            DataFormatParserRegistry.default().parse(
                DataFormat.CSV,
                "a,b\n1,2,3"
            )
        }
    }
}
