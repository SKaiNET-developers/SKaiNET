package sk.ainet.apps.onnx.tools

import kotlin.test.Test
import kotlin.test.assertEquals
import onnx.TensorProto
import pbandk.ByteArr
import java.nio.ByteBuffer
import java.nio.ByteOrder

class WeightExportTest {

    @Test
    fun `extracts float tensor from raw data`() {
        val raw = ByteBuffer.allocate(2 * Float.SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
            .putFloat(1.5f)
            .putFloat(-2.0f)
            .array()
        val tensor = TensorProto(
            name = "weight0",
            dataType = TensorProto.DataType.FLOAT.value,
            dims = listOf(2L),
            rawData = ByteArr(raw)
        )

        val values = tensor.extractValuesAsDoubles()

        assertEquals(listOf(1.5, -2.0), values)
    }

    @Test
    fun `classifies bias and weight names`() {
        val weight = TensorProto(name = "dense.weight", dataType = TensorProto.DataType.FLOAT.value)
        val bias = TensorProto(name = "dense_bias", dataType = TensorProto.DataType.FLOAT.value)
        val other = TensorProto(name = "gamma", dataType = TensorProto.DataType.FLOAT.value)

        assertEquals("weight", weight.parameterKind())
        assertEquals("bias", bias.parameterKind())
        assertEquals("parameter", other.parameterKind())
    }
}
