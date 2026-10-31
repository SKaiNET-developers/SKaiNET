package sk.ainet.io.onnx

import onnx.TensorProto
import sk.ainet.io.ModelReader
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.TensorInfo
import sk.ainet.io.openRandomAccessSource
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.DenseIntArrayTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32

/**
 * [ModelReader] over an ONNX file with streaming (lazy) tensor loading.
 *
 * Opening the reader parses only the model metadata and the initializer *offsets* via
 * [StreamingOnnxReader] / [ProtobufWireReader] — memory stays proportional to the metadata,
 * not the file size. Each [loadTensor] call then reads exactly that tensor's `raw_data`
 * bytes from the underlying [RandomAccessSource].
 *
 * Supported initializer dtypes: `FLOAT` (decoded to a [DenseFloatArrayTensorData]), `INT32`
 * and `INT64` (decoded to a [DenseIntArrayTensorData]; `INT64` values must fit in an `Int`).
 * Initializers stored as typed arrays (`float_data`, …) or in `external_data` sibling files
 * are listed in [tensors] (see [TensorInfo.extra]) but fail loudly in [loadTensor].
 */
public class OnnxModelReader private constructor(
    private val reader: StreamingOnnxReader,
) : ModelReader {

    override val metadata: Map<String, Any> = buildMap {
        put("format", "onnx")
        put("ir_version", reader.irVersion)
        put("model_version", reader.modelVersion)
        if (reader.producerName.isNotEmpty()) put("producer_name", reader.producerName)
        if (reader.producerVersion.isNotEmpty()) put("producer_version", reader.producerVersion)
        if (reader.domain.isNotEmpty()) put("domain", reader.domain)
        if (reader.docString.isNotEmpty()) put("doc_string", reader.docString)
        if (reader.graphName.isNotEmpty()) put("graph_name", reader.graphName)
        for ((opsetDomain, version) in reader.opsetImports) {
            put("opset.$opsetDomain", version)
        }
    }

    override val tensors: Map<String, TensorInfo> = reader.tensors.associate { info ->
        info.name to TensorInfo(
            name = info.name,
            shape = info.shape(),
            dtype = info.dataTypeName,
            offset = info.rawDataOffset,
            size = info.estimatedBytesLong,
            extra = buildMap {
                put("onnx.data_type", info.dataType)
                if (info.hasTypedArrayData) put("onnx.typed_array_data", true)
                val external = info.externalLocation
                if (external != null) {
                    put("onnx.external.location", external)
                    put("onnx.external.offset", info.externalOffset)
                    put("onnx.external.length", info.externalLength)
                }
            },
        )
    }

    override suspend fun loadTensor(name: String): TensorData<*, *> {
        val info = reader.tensors.firstOrNull { it.name == name }
            ?: error("Tensor $name not found")
        check(info.externalLocation == null) {
            "Tensor '$name' is stored in the external data file '${info.externalLocation}'; " +
                "external_data loading is not supported by OnnxModelReader yet"
        }
        val bytes = reader.loadTensorData(info)
        val shape = info.shape()
        return when (info.dataType) {
            TensorProto.DataType.FLOAT.value -> {
                val values = bytes.toFloatArrayLE(name)
                requireElementCount(name, shape, values.size)
                DenseFloatArrayTensorData<FP32>(shape, values)
            }
            TensorProto.DataType.INT32.value -> {
                val values = bytes.toIntArrayLE(name)
                requireElementCount(name, shape, values.size)
                DenseIntArrayTensorData<Int32>(shape, values)
            }
            TensorProto.DataType.INT64.value -> {
                val values = bytes.toLongArrayLE(name)
                requireElementCount(name, shape, values.size)
                DenseIntArrayTensorData<Int32>(
                    shape,
                    IntArray(values.size) { i ->
                        val v = values[i]
                        require(v in Int.MIN_VALUE.toLong()..Int.MAX_VALUE.toLong()) {
                            "Tensor '$name' INT64 value $v at index $i does not fit in Int"
                        }
                        v.toInt()
                    },
                )
            }
            else -> error(
                "Unsupported ONNX dtype ${info.dataTypeName} for tensor '$name' " +
                    "(supported: FLOAT, INT32, INT64)"
            )
        }
    }

    override fun close() {
        reader.close()
    }

    private fun StreamingOnnxTensorInfo.shape(): Shape =
        Shape(IntArray(dims.size) { i -> dims[i].toInt() })

    private fun requireElementCount(name: String, shape: Shape, actual: Int) {
        val expected = shape.dimensions.fold(1L) { acc, d -> acc * d }
        require(actual.toLong() == expected) {
            "Tensor '$name' raw_data holds $actual elements but shape " +
                "${shape.dimensions.contentToString()} expects $expected"
        }
    }

    private fun ByteArray.toFloatArrayLE(name: String): FloatArray {
        require(size % 4 == 0) { "Tensor '$name' raw_data length $size is not divisible by 4 for FLOAT" }
        val out = FloatArray(size / 4)
        for (o in out.indices) {
            out[o] = Float.fromBits(readIntLE(o * 4))
        }
        return out
    }

    private fun ByteArray.toIntArrayLE(name: String): IntArray {
        require(size % 4 == 0) { "Tensor '$name' raw_data length $size is not divisible by 4 for INT32" }
        val out = IntArray(size / 4)
        for (o in out.indices) {
            out[o] = readIntLE(o * 4)
        }
        return out
    }

    private fun ByteArray.toLongArrayLE(name: String): LongArray {
        require(size % 8 == 0) { "Tensor '$name' raw_data length $size is not divisible by 8 for INT64" }
        val out = LongArray(size / 8)
        for (o in out.indices) {
            val lo = readIntLE(o * 8).toLong() and 0xFFFFFFFFL
            val hi = readIntLE(o * 8 + 4).toLong()
            out[o] = (hi shl 32) or lo
        }
        return out
    }

    private fun ByteArray.readIntLE(i: Int): Int =
        (this[i].toInt() and 0xFF) or
            ((this[i + 1].toInt() and 0xFF) shl 8) or
            ((this[i + 2].toInt() and 0xFF) shl 16) or
            ((this[i + 3].toInt() and 0xFF) shl 24)

    public companion object {
        /**
         * Open [source] for streaming access. Metadata and tensor offsets parse now;
         * tensor data loads lazily, per [loadTensor] call. Closing the reader closes [source].
         */
        public fun open(source: RandomAccessSource): OnnxModelReader =
            OnnxModelReader(StreamingOnnxReader.open(source))

        /** Open the ONNX file at [filePath], or `null` where the platform has no file system. */
        public fun open(filePath: String): OnnxModelReader? =
            openRandomAccessSource(filePath)?.let { open(it) }
    }
}
