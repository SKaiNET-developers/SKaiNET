package sk.ainet.apps.kllama

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.models.llama.LlamaLayerWeights
import sk.ainet.models.llama.MemSegWeightConverter
import sk.ainet.models.llama.LlamaModelMetadata
import sk.ainet.models.llama.LlamaRuntimeWeights
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.FloatArrayTensorData
import sk.ainet.lang.tensor.data.Q4MemorySegmentMarker
import sk.ainet.lang.tensor.data.Q8MemorySegmentMarker
import sk.ainet.lang.types.FP32
import java.lang.foreign.Arena

class MemSegWeightConverterTest {

    private val ctx = DirectCpuExecutionContext()
    private val dim = 32       // must be multiple of block size (32)
    private val hiddenDim = 64 // must be multiple of block size (32)
    private val vocab = 4
    private val seqLen = 8

    private val metadata = LlamaModelMetadata(
        architecture = "llama",
        embeddingLength = dim,
        contextLength = seqLen,
        blockCount = 1,
        headCount = 1,
        kvHeadCount = 1,
        feedForwardLength = hiddenDim,
        ropeDimensionCount = dim,
        vocabSize = vocab
    )

    /** Create a raw-byte tensor simulating NATIVE_OPTIMIZED Q4_0 loading. */
    private fun rawQ4Tensor(rows: Int, cols: Int): Pair<sk.ainet.lang.tensor.Tensor<FP32, Float>, Int> {
        val nElements = rows * cols
        val blockSize = 32
        val bytesPerBlock = 18
        val nBlocks = nElements / blockSize
        val nBytes = nBlocks * bytesPerBlock

        // Create Q4_0 packed bytes: each block has 2-byte scale + 16 packed nibbles
        val bytes = ByteArray(nBytes)
        for (block in 0 until nBlocks) {
            val off = block * bytesPerBlock
            // f16 scale = 0.5 → encode as half-float
            val halfBits = floatToHalf(0.5f)
            bytes[off] = (halfBits and 0xFF).toByte()
            bytes[off + 1] = ((halfBits shr 8) and 0xFF).toByte()
            // Nibble codes: all 8 (zero offset) for simplicity
            for (i in 0 until 16) {
                bytes[off + 2 + i] = 0x88.toByte() // hi=8, lo=8
            }
        }

        // Store as IntArray (simulating fromByteArray with Int8 → DenseIntArrayTensorData)
        val byteShape = Shape(nBytes)
        val tensor = ctx.fromByteArray<sk.ainet.lang.types.Int8, Byte>(byteShape, sk.ainet.lang.types.Int8::class, bytes)
        @Suppress("UNCHECKED_CAST")
        return (tensor as sk.ainet.lang.tensor.Tensor<FP32, Float>) to nBytes
    }

    /** Create a raw-byte tensor simulating NATIVE_OPTIMIZED Q8_0 loading. */
    private fun rawQ8Tensor(rows: Int, cols: Int): Pair<sk.ainet.lang.tensor.Tensor<FP32, Float>, Int> {
        val nElements = rows * cols
        val blockSize = 32
        val bytesPerBlock = 34
        val nBlocks = nElements / blockSize
        val nBytes = nBlocks * bytesPerBlock

        val bytes = ByteArray(nBytes)
        for (block in 0 until nBlocks) {
            val off = block * bytesPerBlock
            val halfBits = floatToHalf(0.25f)
            bytes[off] = (halfBits and 0xFF).toByte()
            bytes[off + 1] = ((halfBits shr 8) and 0xFF).toByte()
            // Int8 codes: all zeros
            for (i in 0 until 32) {
                bytes[off + 2 + i] = 0
            }
        }

        val byteShape = Shape(nBytes)
        val tensor = ctx.fromByteArray<sk.ainet.lang.types.Int8, Byte>(byteShape, sk.ainet.lang.types.Int8::class, bytes)
        @Suppress("UNCHECKED_CAST")
        return (tensor as sk.ainet.lang.tensor.Tensor<FP32, Float>) to nBytes
    }

    private fun fpTensor(shape: Shape, value: Float = 1f): sk.ainet.lang.tensor.Tensor<FP32, Float> {
        return ctx.full<FP32, Float>(shape, FP32::class, value)
    }

    private fun floatToHalf(f: Float): Int {
        val bits = f.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        val exp = ((bits ushr 23) and 0xFF) - 127 + 15
        val mant = bits and 0x7FFFFF
        if (exp <= 0) return sign
        if (exp >= 31) return sign or 0x7C00
        return sign or (exp shl 10) or (mant ushr 13)
    }

    private fun createWeightsWithQ4(): LlamaRuntimeWeights<FP32> {
        val kvDim = dim // kvHeadCount == headCount == 1

        val (wq, _) = rawQ4Tensor(dim, dim)
        val (wk, _) = rawQ4Tensor(kvDim, dim)
        val (wv, _) = rawQ4Tensor(kvDim, dim)
        val (wo, _) = rawQ4Tensor(dim, dim)
        val (ffnGate, _) = rawQ4Tensor(hiddenDim, dim)
        val (ffnDown, _) = rawQ4Tensor(dim, hiddenDim)
        val (ffnUp, _) = rawQ4Tensor(hiddenDim, dim)

        val layer = LlamaLayerWeights(
            attnNorm = fpTensor(Shape(dim)),
            wq = wq, wk = wk, wv = wv, wo = wo,
            ffnNorm = fpTensor(Shape(dim)),
            ffnGate = ffnGate, ffnDown = ffnDown, ffnUp = ffnUp
        )

        val quantTypes = mapOf(
            "blk.0.attn_q.weight" to GGMLQuantizationType.Q4_0,
            "blk.0.attn_k.weight" to GGMLQuantizationType.Q4_0,
            "blk.0.attn_v.weight" to GGMLQuantizationType.Q4_0,
            "blk.0.attn_output.weight" to GGMLQuantizationType.Q4_0,
            "blk.0.ffn_gate.weight" to GGMLQuantizationType.Q4_0,
            "blk.0.ffn_down.weight" to GGMLQuantizationType.Q4_0,
            "blk.0.ffn_up.weight" to GGMLQuantizationType.Q4_0,
        )

        return LlamaRuntimeWeights(
            metadata = metadata,
            tokenEmbedding = fpTensor(Shape(vocab, dim)),
            ropeFreqReal = null,
            ropeFreqImag = null,
            layers = listOf(layer),
            outputNorm = fpTensor(Shape(dim)),
            outputWeight = fpTensor(Shape(vocab, dim)),
            quantTypes = quantTypes
        )
    }

    private fun createWeightsWithQ8(): LlamaRuntimeWeights<FP32> {
        val kvDim = dim

        val (wq, _) = rawQ8Tensor(dim, dim)
        val (wk, _) = rawQ8Tensor(kvDim, dim)
        val (wv, _) = rawQ8Tensor(kvDim, dim)
        val (wo, _) = rawQ8Tensor(dim, dim)
        val (ffnGate, _) = rawQ8Tensor(hiddenDim, dim)
        val (ffnDown, _) = rawQ8Tensor(dim, hiddenDim)
        val (ffnUp, _) = rawQ8Tensor(hiddenDim, dim)

        val layer = LlamaLayerWeights(
            attnNorm = fpTensor(Shape(dim)),
            wq = wq, wk = wk, wv = wv, wo = wo,
            ffnNorm = fpTensor(Shape(dim)),
            ffnGate = ffnGate, ffnDown = ffnDown, ffnUp = ffnUp
        )

        val quantTypes = mapOf(
            "blk.0.attn_q.weight" to GGMLQuantizationType.Q8_0,
            "blk.0.attn_k.weight" to GGMLQuantizationType.Q8_0,
            "blk.0.attn_v.weight" to GGMLQuantizationType.Q8_0,
            "blk.0.attn_output.weight" to GGMLQuantizationType.Q8_0,
            "blk.0.ffn_gate.weight" to GGMLQuantizationType.Q8_0,
            "blk.0.ffn_down.weight" to GGMLQuantizationType.Q8_0,
            "blk.0.ffn_up.weight" to GGMLQuantizationType.Q8_0,
        )

        return LlamaRuntimeWeights(
            metadata = metadata,
            tokenEmbedding = fpTensor(Shape(vocab, dim)),
            ropeFreqReal = null,
            ropeFreqImag = null,
            layers = listOf(layer),
            outputNorm = fpTensor(Shape(dim)),
            outputWeight = fpTensor(Shape(vocab, dim)),
            quantTypes = quantTypes
        )
    }

    // ── Tests ───────────────────────────────────────────────────────────────

    @Test
    fun `convert with empty quantTypes returns same weights`() {
        val weights = LlamaRuntimeWeights(
            metadata = metadata,
            tokenEmbedding = fpTensor(Shape(vocab, dim)),
            ropeFreqReal = null,
            ropeFreqImag = null,
            layers = emptyList(),
            outputNorm = fpTensor(Shape(dim)),
            outputWeight = fpTensor(Shape(vocab, dim)),
            quantTypes = emptyMap()
        )
        val arena = Arena.ofConfined()
        val result = MemSegWeightConverter.convert(weights, ctx, arena)
        assertSame(weights, result, "Should return same instance when no quant types")
        arena.close()
    }

    @Test
    fun `convert Q4_0 layer weights to Q4 MemorySegment`() {
        val arena = Arena.ofConfined()
        val weights = createWeightsWithQ4()

        val converted = MemSegWeightConverter.convert(weights, ctx, arena)

        val layer = converted.layers[0]
        assertTrue(layer.wq.data is Q4MemorySegmentMarker, "wq should be Q4 MemorySegment")
        assertTrue(layer.wk.data is Q4MemorySegmentMarker, "wk should be Q4 MemorySegment")
        assertTrue(layer.wv.data is Q4MemorySegmentMarker, "wv should be Q4 MemorySegment")
        assertTrue(layer.wo.data is Q4MemorySegmentMarker, "wo should be Q4 MemorySegment")
        assertTrue(layer.ffnGate.data is Q4MemorySegmentMarker, "ffnGate should be Q4 MemorySegment")
        assertTrue(layer.ffnDown.data is Q4MemorySegmentMarker, "ffnDown should be Q4 MemorySegment")
        assertTrue(layer.ffnUp.data is Q4MemorySegmentMarker, "ffnUp should be Q4 MemorySegment")
        arena.close()
    }

    @Test
    fun `convert Q8_0 layer weights to Q8 MemorySegment`() {
        val arena = Arena.ofConfined()
        val weights = createWeightsWithQ8()

        val converted = MemSegWeightConverter.convert(weights, ctx, arena)

        val layer = converted.layers[0]
        assertTrue(layer.wq.data is Q8MemorySegmentMarker, "wq should be Q8 MemorySegment")
        assertTrue(layer.wk.data is Q8MemorySegmentMarker, "wk should be Q8 MemorySegment")
        assertTrue(layer.wv.data is Q8MemorySegmentMarker, "wv should be Q8 MemorySegment")
        assertTrue(layer.wo.data is Q8MemorySegmentMarker, "wo should be Q8 MemorySegment")
        assertTrue(layer.ffnGate.data is Q8MemorySegmentMarker, "ffnGate should be Q8 MemorySegment")
        assertTrue(layer.ffnDown.data is Q8MemorySegmentMarker, "ffnDown should be Q8 MemorySegment")
        assertTrue(layer.ffnUp.data is Q8MemorySegmentMarker, "ffnUp should be Q8 MemorySegment")
        arena.close()
    }

    @Test
    fun `convert preserves float tensors unchanged`() {
        val arena = Arena.ofConfined()
        val weights = createWeightsWithQ4()

        val converted = MemSegWeightConverter.convert(weights, ctx, arena)

        // Norms and embeddings should remain as FloatArrayTensorData (not converted)
        val layer = converted.layers[0]
        assertTrue(layer.attnNorm.data is FloatArrayTensorData<*>,
            "attnNorm should remain float: ${layer.attnNorm.data::class.simpleName}")
        assertTrue(layer.ffnNorm.data is FloatArrayTensorData<*>,
            "ffnNorm should remain float: ${layer.ffnNorm.data::class.simpleName}")
        assertTrue(converted.outputNorm.data is FloatArrayTensorData<*>,
            "outputNorm should remain float")
        arena.close()
    }

    @Test
    fun `converted Q4 tensors have correct logical shape`() {
        val arena = Arena.ofConfined()
        val weights = createWeightsWithQ4()

        val converted = MemSegWeightConverter.convert(weights, ctx, arena)
        val layer = converted.layers[0]

        assertEquals(Shape(dim, dim), layer.wq.shape, "wq shape")
        assertEquals(Shape(dim, dim), layer.wk.shape, "wk shape")
        assertEquals(Shape(dim, dim), layer.wv.shape, "wv shape")
        assertEquals(Shape(dim, dim), layer.wo.shape, "wo shape")
        assertEquals(Shape(hiddenDim, dim), layer.ffnGate.shape, "ffnGate shape")
        assertEquals(Shape(dim, hiddenDim), layer.ffnDown.shape, "ffnDown shape")
        assertEquals(Shape(hiddenDim, dim), layer.ffnUp.shape, "ffnUp shape")
        arena.close()
    }

    @Test
    fun `converted Q8 tensors have correct logical shape`() {
        val arena = Arena.ofConfined()
        val weights = createWeightsWithQ8()

        val converted = MemSegWeightConverter.convert(weights, ctx, arena)
        val layer = converted.layers[0]

        assertEquals(Shape(dim, dim), layer.wq.shape, "wq shape")
        assertEquals(Shape(hiddenDim, dim), layer.ffnGate.shape, "ffnGate shape")
        assertEquals(Shape(dim, hiddenDim), layer.ffnDown.shape, "ffnDown shape")
        arena.close()
    }

    @Test
    fun `non-quantized token embedding passes through`() {
        val arena = Arena.ofConfined()
        val weights = createWeightsWithQ4()

        val converted = MemSegWeightConverter.convert(weights, ctx, arena)

        // tokenEmbedding is not in quantTypes, should pass through unchanged
        assertTrue(converted.tokenEmbedding.data is FloatArrayTensorData<*>,
            "tokenEmbedding should remain float when not quantized")
        assertEquals(Shape(vocab, dim), converted.tokenEmbedding.shape)
        arena.close()
    }
}
