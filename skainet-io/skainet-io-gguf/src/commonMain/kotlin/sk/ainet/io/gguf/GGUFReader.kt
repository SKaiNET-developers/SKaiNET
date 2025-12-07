package sk.ainet.io.gguf

import kotlinx.io.Source
import kotlinx.io.readByteArray
import sk.ainet.io.gguf.utils.Endian
import sk.ainet.io.gguf.utils.numberOfBytes
import sk.ainet.io.gguf.utils.readDataByType
import kotlin.math.pow
import kotlin.reflect.KClass

/**
 * This is a kotlin gguf reader interpreted from python code "gguf-py/gguf/gguf_reader.py"
 * of github repo "https://github.com/ggerganov/llama.cpp"
 */

// Constants
val READER_SUPPORTED_VERSIONS = listOf(2, GGUF_VERSION)

// Data classes for ReaderField and ReaderTensor
data class ReaderField(
    val offset: Int,                 // Offset to start of this field
    val name: String,                // Name of the field (not necessarily from file data)
    /*
     Data parts. Some types have multiple components, such as strings that consist of a length followed by the string data.
     */
    val parts: List<List<Any>> = emptyList(),
    /*
     Indexes into parts that we can call the actual data. For example an array of strings will be populated with indexes to the actual
     string data.
     */
    val data: List<Int> = listOf(-1),
    val types: List<GGUFValueType> = emptyList() // Data types corresponding to parts
)

data class ReaderTensor(
    val name: String,
    val tensorType: GGMLQuantizationType,
    val shape: List<UInt>,
    val nElements: Int,
    val nBytes: Int,
    val dataOffset: Int,
    val data: List<Any>,
    val field: ReaderField
)

// Data class to hold the return values
data class FieldParts(
    val size: Int,
    val parts: List<List<Any>>,
    val idxs: List<Int>,
    val types: List<GGUFValueType>
)

@OptIn(ExperimentalUnsignedTypes::class)
class GGUFReader(
    source: Source,
    private val loadTensorData: Boolean = true,
    decodeF16ToFloat: Boolean = true,
    decodeBF16ToFloat: Boolean = true
) {
    // Public API additions
    /**
     * Lazily materialize the raw payload for the given tensor, honoring its ggml quantization.
     * Returns a typed Kotlin List corresponding to the underlying storage:
     * - F32 -> List<Float>, F64 -> List<Double>, I8 -> List<Byte>, I16 -> List<Short>, I32 -> List<Int>, I64 -> List<Long>
     * - Quantized types -> List<UByte> of raw bytes with shape adapted via quantShapeToByteShape
     */
    fun materialize(tensor: ReaderTensor): List<Any> = materializeTensorData(
        ggmlType = tensor.tensorType,
        dataOffs = tensor.dataOffset,
        nElems = tensor.nElements,
        nBytes = tensor.nBytes
    )
    // Properties
    var byteOrder: Char = 'I' // 'I' - same as host, 'S' - swapped
    var alignment: Int = GGUF_DEFAULT_ALIGNMENT
    var dataOffset: Int = 0
    val fields: LinkedHashMap<String, ReaderField> = linkedMapOf()
    var tensors: MutableList<ReaderTensor> = mutableListOf()
    /** Toggle decoding of F16 payloads into Float values; if false, preserves UShort raw words. */
    var decodeF16ToFloat: Boolean = decodeF16ToFloat
    /** Toggle decoding of BF16 payloads into Float values; if false, preserves UShort raw words. */
    var decodeBF16ToFloat: Boolean = decodeBF16ToFloat

    private val data: ByteArray
    private var offs = 0
    private var tensorCount: ULong = 0u

    // Mapping GGUFValueType to Kotlin types (or placeholders for illustrative purposes)
    val ggufScalarToKotlinType: Map<GGUFValueType, KClass<*>> = mapOf(
        GGUFValueType.UINT8 to UByte::class,
        GGUFValueType.INT8 to Byte::class,
        GGUFValueType.UINT16 to UShort::class,
        GGUFValueType.INT16 to Short::class,
        GGUFValueType.UINT32 to UInt::class,
        GGUFValueType.INT32 to Int::class,
        GGUFValueType.FLOAT32 to Float::class,
        GGUFValueType.UINT64 to ULong::class,
        GGUFValueType.INT64 to Long::class,
        GGUFValueType.FLOAT64 to Double::class,
        GGUFValueType.BOOL to Boolean::class
    )

    init {
        data = source.readByteArray()

        checkGGUFMagicNumber().then {
            checkGGUFVersion().then {
                checkTensorAndKvCounts().then {
                    buildTensorInfoFields()
                }
            }
        }
    }

    private fun Unit.then(function: () -> Unit) {
        function()
    }

    // Internal: materialize raw tensor payload based on ggml type
    private fun materializeTensorData(
        ggmlType: GGMLQuantizationType,
        dataOffs: Int,
        nElems: Int,
        nBytes: Int
    ): List<Any> {
        return when (ggmlType) {
            GGMLQuantizationType.F16 -> data.readDataByType<UShort>(dataOffs, nElems).let { halfs ->
                if (decodeF16ToFloat) halfs.map { halfToFloat(it) } else halfs
            }
            GGMLQuantizationType.BF16 -> data.readDataByType<UShort>(dataOffs, nElems).let { bf16s ->
                if (decodeBF16ToFloat) bf16s.map { bfloat16ToFloat(it) } else bf16s
            }
            GGMLQuantizationType.F32 -> data.readDataByType<Float>(dataOffs, nElems)
            GGMLQuantizationType.F64 -> data.readDataByType<Double>(dataOffs, nElems)
            GGMLQuantizationType.I8 -> data.readDataByType<Byte>(dataOffs, nElems)
            GGMLQuantizationType.I16 -> data.readDataByType<Short>(dataOffs, nElems)
            GGMLQuantizationType.I32 -> data.readDataByType<Int>(dataOffs, nElems)
            GGMLQuantizationType.I64 -> data.readDataByType<Long>(dataOffs, nElems)
            else -> data.readDataByType<UByte>(dataOffs, nBytes)
        }
    }

    private fun buildTensorInfoFields() {
        // Build tensor info fields
        val (newOffs, tensorFields) = buildTensorInfo(offs, tensorCount.toInt())
        offs = newOffs
        val newAlign = fields["general.alignment"]
        if (newAlign != null && newAlign.types == listOf(GGUFValueType.UINT32)) {
            alignment = newAlign.parts.last()[0] as Int
        }
        val padding = offs % alignment
        if (padding != 0) {
            offs += alignment - padding
        }
        dataOffset = offs
        buildTensors(offs, tensorFields)
    }

    private fun checkTensorAndKvCounts() {
        val tempCounts = data.readDataByType<ULong>(offs, 2)
        offs += pushField(
            ReaderField(
                offs,
                "GGUF.tensor_count",
                listOf(tempCounts.take(1)),
                listOf(0),
                listOf(GGUFValueType.UINT64)
            )
        )
        offs += pushField(
            ReaderField(
                offs,
                "GGUF.kv_count",
                listOf(tempCounts.drop(1)),
                listOf(0),
                listOf(GGUFValueType.UINT64)
            )
        )
        tensorCount = tempCounts[0]
        val kvCount = tempCounts[1]
        offs = buildFields(offs, kvCount.toInt())
    }

    private fun checkGGUFVersion() {
        val version = data.readDataByType<UInt>(offs, 1)[0]
        if (version.toInt() !in READER_SUPPORTED_VERSIONS) {
            throw IllegalArgumentException("Unsupported GGUF version: $version")
        }
        fields.clear()
        tensors.clear()
        offs += pushField(
            ReaderField(
                offs,
                "GGUF.version",
                listOf(listOf(version)),
                listOf(0),
                listOf(GGUFValueType.UINT32)
            )
        )
    }

    private fun checkGGUFMagicNumber() {
        val magicNumber = data.readDataByType<UInt>(0, 1, Endian.LITTLE_ENDIAN)[0]
        if (magicNumber != GGUF_MAGIC) {
            throw IllegalArgumentException("GGUF magic invalid")
        }
        offs += 4
    }


    private fun pushField(field: ReaderField, skipSum: Boolean = false): Int {
        if (fields.contains(field.name)) {
            // TODO: add option to generate error on duplicate keys
            // raise KeyError(f'Duplicate {field.name} already in list at offset {field.offset}')
            fields["${field.name}_${field.offset}"] = field

        } else {
            fields[field.name] = field
        }
        return if (skipSum) 0 else field.parts.sumOf { it.numberOfBytes() }
    }

    private inline fun <reified T> ReaderField.partAs(index: Int): List<T> {
        val part = parts.getOrNull(index)
            ?: throw IllegalArgumentException("Expected part at index $index for field '$name'")
        return part.mapIndexed { idx, value ->
            require(value is T) {
                "Unexpected type in field '$name' part $index at element $idx: ${value::class}, expected ${T::class}"
            }
            value
        }
    }

    private fun getStr(offset: Int): Pair<List<ULong>, List<UByte>> {
        val slen = data.readDataByType<ULong>(offset, 1)
        val second = data.readDataByType<UByte>(offset + 8, slen[0].toInt())

        return Pair(slen, second)
    }

    private fun getFieldParts(origOffs: Int, rawType: Int): FieldParts {
        var offs = origOffs
        val types = mutableListOf<GGUFValueType>()
        val gtype = GGUFValueType.entries.find { it.value == rawType }
            ?: throw IllegalArgumentException("GGUFValueType $rawType not defined")
        types.add(gtype)

        // Handle strings.
        if (gtype == GGUFValueType.STRING) {
            val sparts = listOf(getStr(offs).first, getStr(offs).second)
            val size = sparts.sumOf { it.numberOfBytes() }
            return FieldParts(size, sparts, listOf(1), types)
        }

        // Check if it's a simple scalar type.
        val nptype = ggufScalarToKotlinType[gtype]

        if (nptype != null) {
            val value = when (nptype) {
                UByte::class -> data.readDataByType<UByte>(offs)
                Byte::class -> data.readDataByType<Byte>(offs)
                UShort::class -> data.readDataByType<UShort>(offs)
                Short::class -> data.readDataByType<Short>(offs)
                UInt::class -> data.readDataByType<UInt>(offs)
                Int::class -> data.readDataByType<Int>(offs)
                Float::class -> data.readDataByType<Float>(offs)
                ULong::class -> data.readDataByType<ULong>(offs)
                Long::class -> data.readDataByType<Long>(offs)
                Double::class -> data.readDataByType<Double>(offs)
                Boolean::class -> data.readDataByType<Boolean>(offs)
                else -> throw IllegalArgumentException("getFieldParts: nptype $nptype not supported")
            }
            return FieldParts(value.numberOfBytes(), listOf(value), listOf(0), types)
        }

        // Handle arrays.
        if (gtype == GGUFValueType.ARRAY) {
            val rawItype = data.readDataByType<UInt>(offs)
            offs += rawItype.numberOfBytes()
            val alen = data.readDataByType<ULong>(offs)
            offs += alen.numberOfBytes()
            val aparts: MutableList<List<Any>> = mutableListOf(rawItype, alen)
            val dataIdxs = mutableListOf<Int>()

            for (idx in 0 until alen[0].toInt()) {
                val temp = getFieldParts(offs, rawItype[0].toInt())
                val currSize = temp.size
                val currParts = temp.parts
                val currIdxs = temp.idxs
                val currTypes = temp.types
                if (idx == 0) {
                    types.addAll(currTypes)
                }
                val idxsOffs = aparts.size
                aparts.addAll(currParts)
                dataIdxs.addAll(currIdxs.map { it + idxsOffs })
                offs += currSize
            }
            return FieldParts(offs - origOffs, aparts, dataIdxs, types)
        }

        // We can't deal with this one.
        throw IllegalArgumentException("Unknown/unhandled field type $gtype")
    }

    private fun getTensorInfoField(origOffs: Int): ReaderField {
        var offs = origOffs

        // Get Tensor Name
        val (nameLen, nameData) = getStr(offs)
        offs += nameLen.numberOfBytes() + nameData.numberOfBytes()

        // Get Tensor Dimensions Count
        val nDims = data.readDataByType<UInt>(offs)
        offs += nDims.numberOfBytes()

        // Get Tensor Dimension Array
        val dims = data.readDataByType<ULong>(offs, nDims[0].toInt())
        offs += dims.numberOfBytes()

        // Get Tensor Encoding Scheme Type
        val rawDtype = data.readDataByType<UInt>(offs)
        offs += rawDtype.numberOfBytes()

        // Get Tensor Offset
        val offsetTensor = data.readDataByType<ULong>(offs)
        offs += offsetTensor.numberOfBytes()

        val utf8String: String = nameData.toUByteArray().toByteArray().decodeToString()

        return ReaderField(
            origOffs,
            utf8String,
            listOf(nameLen, nameData, nDims, dims, rawDtype, offsetTensor),
            listOf(1, 3, 4, 5)
        )
    }

    private fun buildFields(offs: Int, count: Int): Int {
        var currentOffs = offs
        for (i in 0 until count) {
            val origOffs = currentOffs
            val (kvKlen, kvKdata) = getStr(currentOffs)
            currentOffs += kvKlen.numberOfBytes() + kvKdata.numberOfBytes()
            val rawKvType = data.readDataByType<UInt>(currentOffs)
            currentOffs += rawKvType.numberOfBytes()
            val parts: MutableList<List<Any>> = mutableListOf(kvKlen, kvKdata, rawKvType)
            val idxsOffs = parts.size
            val temp = getFieldParts(currentOffs, rawKvType[0].toInt())
            val fieldSize = temp.size
            val fieldParts = temp.parts
            val fieldIdxs = temp.idxs
            val fieldTypes = temp.types

            val kvKdataUtf8String: String = kvKdata.toUByteArray().toByteArray().decodeToString()


            parts.addAll(fieldParts)
            pushField(
                ReaderField(
                    offset = origOffs,
                    name = kvKdataUtf8String,
                    parts = parts,
                    data = fieldIdxs.map { it + idxsOffs },
                    types = fieldTypes
                ),
                skipSum = true
            )
            currentOffs += fieldSize
        }
        return currentOffs
    }


    private fun buildTensorInfo(offs: Int, count: Int): Pair<Int, List<ReaderField>> {
        val tensorFields = mutableListOf<ReaderField>()
        var currentOffs = offs
        repeat(count) {
            val field = getTensorInfoField(currentOffs)
            currentOffs += field.parts.sumOf { it.numberOfBytes() }
            tensorFields.add(field)
        }
        return Pair(currentOffs, tensorFields)
    }

    private fun buildTensors(startOffs: Int, fields: List<ReaderField>) {
        val tensors = mutableListOf<ReaderTensor>()
        val tensorNames = mutableSetOf<String>() // keep track of names to prevent duplicate tensors

        for (field in fields) {
            val _nameLen = field.partAs<ULong>(0)
            val nameData = field.partAs<UByte>(1)
            val _nDims = field.partAs<UInt>(2)
            val dims = field.partAs<ULong>(3)
            val rawDtype = field.partAs<UInt>(4)
            val offsetTensor = field.partAs<ULong>(5)

            val tensorName: String = nameData.toUByteArray().toByteArray().decodeToString()
            if (tensorNames.contains(tensorName)) {
                throw IllegalArgumentException("buildTensors: Found duplicated tensor with name $tensorName")
            }
            tensorNames.add(tensorName)

            val ggmlType = GGMLQuantizationType.fromValue(rawDtype[0].toInt())
                ?: throw IllegalArgumentException("Invalid ggmlType")
            val nElems = dims.reduce { acc, dim -> acc * dim }
            var npDims = dims.reversed()
            val (blockSize, typeSize) = GGML_QUANT_SIZES[ggmlType]
                ?: throw IllegalArgumentException("Invalid quantization type")
            val nBytes = nElems.toInt() * typeSize / blockSize
            val dataOffs = startOffs + offsetTensor[0].toInt()

            // For non-native/quantized types, tensor payload is stored as bytes
            if (ggmlType !in listOf(
                    GGMLQuantizationType.F32,
                    GGMLQuantizationType.F64,
                    GGMLQuantizationType.I8,
                    GGMLQuantizationType.I16,
                    GGMLQuantizationType.I32,
                    GGMLQuantizationType.I64
                )
            ) {
                npDims = quantShapeToByteShape(npDims, ggmlType)
            }

            val materializedData: List<Any> = if (loadTensorData) {
                materializeTensorData(
                    ggmlType = ggmlType,
                    dataOffs = dataOffs,
                    nElems = nElems.toInt(),
                    nBytes = nBytes
                )
            } else {
                emptyList()
            }

            tensors.add(
                ReaderTensor(
                    name = tensorName,
                    tensorType = ggmlType,
                    shape = dims.map { it.toUInt() },
                    nElements = nElems.toInt(),
                    nBytes = nBytes,
                    dataOffset = dataOffs,
                    data = materializedData,
                    field = field
                )
            )
        }
        this.tensors = tensors
    }

    private fun halfToFloat(bits: UShort): Float {
        val sign = (bits.toInt() shr 15) and 0x1
        val exp = (bits.toInt() shr 10) and 0x1F
        val mant = bits.toInt() and 0x3FF

        val value = when (exp) {
            0 -> mant * 2.0.pow(-24.0)
            31 -> if (mant == 0) Double.POSITIVE_INFINITY else Double.NaN
            else -> (1 + mant / 1024.0) * 2.0.pow(exp - 15)
        }
        return if (sign == 1) (-value).toFloat() else value.toFloat()
    }

    private fun bfloat16ToFloat(bits: UShort): Float {
        val shifted = bits.toInt() shl 16
        return Float.fromBits(shifted)
    }
}
