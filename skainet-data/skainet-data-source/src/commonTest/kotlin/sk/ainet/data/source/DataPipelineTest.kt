package sk.ainet.data.source

import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class DataPipelineTest {
    @Test
    fun executesTypedStagesInOrder() = runTest {
        val pipeline = dataPipeline<Int>()
            .stage(stage<Int, Int>("add-one") { input -> input + 1 })
            .stage(stage<Int, String>("stringify") { input -> "value=$input" })

        assertEquals("add-one -> stringify", pipeline.describe())
        assertEquals(listOf("add-one", "stringify"), pipeline.stageNames)
        assertEquals("value=42", pipeline.execute(41))
    }

    @Test
    fun identityPipelineReturnsInput() = runTest {
        val pipeline = dataPipeline<Int>()

        assertEquals("", pipeline.describe())
        assertEquals(7, pipeline.execute(7))
    }

    @Test
    fun rejectsInvalidStageInput() = runTest {
        val pipeline = dataPipeline<Int>()
            .stage(
                stage<Int, Int>(
                    name = "positive",
                    validate = { input -> input > 0 }
                ) { input -> input }
            )

        assertFailsWith<DataPipelineException> {
            pipeline.execute(0)
        }
    }

    @Test
    fun transformerUpdatesSchemaAndRows() = runTest {
        val dropLabel = dataTransformer<RawDataset, RawDataset>(
            name = "drop-label",
            outputSchema = { schema -> DataSchema(schema.columns.filter { column -> column != "label" }) }
        ) { dataset ->
            val columns = dataset.schema.columns.filter { column -> column != "label" }
            dataset.copy(
                schema = DataSchema(columns),
                rows = dataset.rows.map { row ->
                    RawDataRow(row.values.filterKeys { key -> key in columns })
                }
            )
        }
        val input = RawDataset(
            rows = listOf(RawDataRow(mapOf("id" to "1", "label" to "cat"))),
            schema = DataSchema(listOf("id", "label"))
        )

        val output = (dataPipeline<RawDataset>() then dropLabel).execute(input)

        assertEquals(DataSchema(listOf("id")), dropLabel.getOutputSchema(input.schema))
        assertEquals(DataSchema(listOf("id")), output.schema)
        assertEquals(mapOf("id" to "1"), output.rows.single().values)
    }
}
