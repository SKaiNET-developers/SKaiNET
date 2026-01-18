package sk.ainet.io.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TensorInfoTest {

    @Test
    fun testBasicTensorInfoCreation() {
        val tensor = TensorInfo(
            name = "weight",
            shape = listOf(3L, 224L, 224L),
            dataType = DataType.FLOAT32,
            elementCount = 3L * 224L * 224L,
            sizeInBytes = 3L * 224L * 224L * 4,
            format = ModelFormat.ONNX
        )

        assertEquals("weight", tensor.name)
        assertEquals(listOf(3L, 224L, 224L), tensor.shape)
        assertEquals(DataType.FLOAT32, tensor.dataType)
        assertEquals(150528L, tensor.elementCount)
        assertEquals(602112L, tensor.sizeInBytes)
        assertEquals(ModelFormat.ONNX, tensor.format)
    }

    @Test
    fun testTensorInfoWithNativeDType() {
        val tensor = TensorInfo(
            name = "layer.0.weight",
            shape = listOf(512L, 768L),
            dataType = DataType.FLOAT16,
            elementCount = 512L * 768L,
            sizeInBytes = 512L * 768L * 2,
            format = ModelFormat.GGUF,
            nativeDType = "F16",
            skainetDType = "Float16",
            canLoadNatively = true
        )

        assertEquals("F16", tensor.nativeDType)
        assertEquals("Float16", tensor.skainetDType)
        assertTrue(tensor.canLoadNatively)
    }

    @Test
    fun testShapeString() {
        val tensor = TensorInfo(
            name = "test",
            shape = listOf(1L, 3L, 224L, 224L),
            dataType = DataType.FLOAT32,
            elementCount = 150528L,
            sizeInBytes = 602112L,
            format = ModelFormat.ONNX
        )

        assertEquals("[1, 3, 224, 224]", tensor.shapeString)
    }

    @Test
    fun testShapeStringEmpty() {
        val tensor = TensorInfo(
            name = "scalar",
            shape = emptyList(),
            dataType = DataType.FLOAT32,
            elementCount = 0L,
            sizeInBytes = 0L,
            format = ModelFormat.ONNX
        )

        assertEquals("[]", tensor.shapeString)
    }

    @Test
    fun testSizeStringBytes() {
        val tensor = TensorInfo(
            name = "small",
            shape = listOf(10L),
            dataType = DataType.FLOAT32,
            elementCount = 10L,
            sizeInBytes = 40L,
            format = ModelFormat.ONNX
        )

        assertEquals("40 B", tensor.sizeString)
    }

    @Test
    fun testSizeStringKilobytes() {
        val tensor = TensorInfo(
            name = "medium",
            shape = listOf(1000L),
            dataType = DataType.FLOAT32,
            elementCount = 1000L,
            sizeInBytes = 4000L,
            format = ModelFormat.ONNX
        )

        assertEquals("3 KB", tensor.sizeString)
    }

    @Test
    fun testSizeStringMegabytes() {
        val tensor = TensorInfo(
            name = "large",
            shape = listOf(1000000L),
            dataType = DataType.FLOAT32,
            elementCount = 1000000L,
            sizeInBytes = 4000000L,
            format = ModelFormat.ONNX
        )

        assertTrue(tensor.sizeString.contains("MB"))
    }

    @Test
    fun testSizeStringGigabytes() {
        val tensor = TensorInfo(
            name = "huge",
            shape = listOf(1000000000L),
            dataType = DataType.FLOAT32,
            elementCount = 1000000000L,
            sizeInBytes = 4000000000L,
            format = ModelFormat.ONNX
        )

        assertTrue(tensor.sizeString.contains("GB"))
    }

    @Test
    fun testSizeStringUnknown() {
        val tensor = TensorInfo(
            name = "unknown_size",
            shape = listOf(10L),
            dataType = DataType.STRING,
            elementCount = 10L,
            sizeInBytes = null,
            format = ModelFormat.ONNX
        )

        assertEquals("unknown", tensor.sizeString)
    }

    @Test
    fun testDataClassEquality() {
        val tensor1 = TensorInfo(
            name = "weight",
            shape = listOf(3L, 3L),
            dataType = DataType.FLOAT32,
            elementCount = 9L,
            sizeInBytes = 36L,
            format = ModelFormat.ONNX
        )

        val tensor2 = TensorInfo(
            name = "weight",
            shape = listOf(3L, 3L),
            dataType = DataType.FLOAT32,
            elementCount = 9L,
            sizeInBytes = 36L,
            format = ModelFormat.ONNX
        )

        assertEquals(tensor1, tensor2)
        assertEquals(tensor1.hashCode(), tensor2.hashCode())
    }

    @Test
    fun testDataClassCopy() {
        val original = TensorInfo(
            name = "weight",
            shape = listOf(3L, 3L),
            dataType = DataType.FLOAT32,
            elementCount = 9L,
            sizeInBytes = 36L,
            format = ModelFormat.ONNX
        )

        val modified = original.copy(name = "bias")

        assertEquals("bias", modified.name)
        assertEquals(original.shape, modified.shape)
        assertEquals(original.dataType, modified.dataType)
    }

    @Test
    fun testGgufTensor() {
        val tensor = TensorInfo(
            name = "model.layers.0.attention.q_proj.weight",
            shape = listOf(4096L, 4096L),
            dataType = DataType.UNKNOWN,  // For quantized types
            elementCount = 4096L * 4096L,
            sizeInBytes = null,  // Quantized size varies
            format = ModelFormat.GGUF,
            nativeDType = "Q4_K_M",
            skainetDType = null,
            canLoadNatively = false
        )

        assertEquals(ModelFormat.GGUF, tensor.format)
        assertEquals("Q4_K_M", tensor.nativeDType)
        assertEquals(false, tensor.canLoadNatively)
    }

    @Test
    fun testOnnxTensor() {
        val tensor = TensorInfo(
            name = "input",
            shape = listOf(1L, 3L, 224L, 224L),
            dataType = DataType.FLOAT32,
            elementCount = 150528L,
            sizeInBytes = 602112L,
            format = ModelFormat.ONNX,
            nativeDType = "FLOAT",
            skainetDType = "Float32",
            canLoadNatively = true
        )

        assertEquals(ModelFormat.ONNX, tensor.format)
        assertEquals("FLOAT", tensor.nativeDType)
        assertEquals("Float32", tensor.skainetDType)
        assertTrue(tensor.canLoadNatively)
    }
}
