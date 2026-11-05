package sk.ainet.compile.hlo.converters

import sk.ainet.compile.hlo.ConversionContext
import sk.ainet.compile.hlo.ConversionResult
import sk.ainet.compile.hlo.StableHloOperationConverter
import sk.ainet.compile.opt.passes.BlockQuantMatmulFusionPass
import sk.ainet.lang.graph.GraphNode
import sk.ainet.lang.tensor.data.Q4_0TensorData
import sk.ainet.lang.tensor.data.Q8_0TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.tensor.storage.TensorEncoding

/**
 * Lowers `block_quant_matmul` (made by [BlockQuantMatmulFusionPass]) to a linalg chain that keeps
 * the weight as integer codes plus per-block scales:
 * ```
 *   w[nB, 32, N] = (sitofp(codes) - zp) * scale       linalg.generic, scale/zp broadcast over the block
 *   y[1, M, N]   = batch_matmul(x[1, M, K], expand(collapse(w)))
 * ```
 * Codes are `i8` for Q8_0 and `i4` for Q4_0; scales and the optional zero point are `bf16`
 * `[nB, 1, N]`; all of them are inline constants. The activation is converted to `bf16` and
 * reshaped to `[1, M, K]` around the chain, and the result is brought back to the node's type.
 *
 * This is the form the Torq compiler (2.3.0, `--iree-input-type=stablehlo-torq`) matches to its
 * `mixed_quant_matmul` kernel when `M = 1`. The kernel accepts only positive scales, so a block
 * with a negative scale `d` is stored exactly as codes `-1 - c`, zero point `-1` and scale `-d`;
 * the zero-point tensor is emitted only when such a block exists. The chain is portable linalg,
 * but stock IREE `llvm-cpu` has given wrong results for `i4` weights, so keep Q4_0 for targets
 * that were checked.
 */
public class BlockQuantMatmulConverter : StableHloOperationConverter {

    override val supportedOperations: Set<String> = setOf(BlockQuantMatmulFusionPass.OP_NAME)

    override fun convert(node: GraphNode, operands: List<String>, context: ConversionContext): ConversionResult {
        val storage = node.operation.parameters[BlockQuantMatmulFusionPass.WEIGHT] as? PackedBlockStorage
            ?: return ConversionResult.Failure("block_quant_matmul ${node.id} has no packed weight", "missing weight on ${node.id}")
        if (operands.size != 1) {
            return ConversionResult.Failure("block_quant_matmul takes one operand, got ${operands.size}", "bad arity on ${node.id}")
        }
        val inSpec = node.inputs.firstOrNull()
        val outSpec = node.outputs.firstOrNull()
        val inShape = inSpec?.shape
        val outShape = outSpec?.shape
        if (inShape.isNullOrEmpty() || outShape.isNullOrEmpty()) {
            return ConversionResult.Failure("block_quant_matmul ${node.id} needs static input and output shapes", "unknown shape on ${node.id}")
        }
        val weight = try {
            BlockQuantWeight.of(storage)
        } catch (e: IllegalArgumentException) {
            return ConversionResult.Failure(e.message ?: "unsupported weight", "weight of ${node.id}")
        }
        val k = weight.k
        val n = weight.n
        val m = inShape.dropLast(1).fold(1) { acc, d -> acc * d }
        if (inShape.last() != k || outShape.last() != n || outShape.dropLast(1).fold(1) { acc, d -> acc * d } != m) {
            return ConversionResult.Failure(
                "block_quant_matmul ${node.id}: x $inShape · W[$n, $k]ᵀ does not give $outShape",
                "shape mismatch on ${node.id}"
            )
        }

        val types = context.getTypeMapper()
        val inElem = types.mapDType(inSpec.dtype)
        val outElem = types.mapDType(outSpec.dtype)
        val nB = weight.blockCount
        val b = BlockQuantWeight.BLOCK
        val codeType = "i${weight.bits}"
        fun t(vararg dims: Int, elem: String = "bf16") = "tensor<${dims.joinToString("x")}x$elem>"
        fun tt(dims: List<Int>, elem: String) = "tensor<${dims.joinToString("x")}x$elem>"
        val ops = mutableListOf<String>()
        fun emit(op: String) { context.emitOperation(op); ops += op }

        val weightName = node.operation.parameters[BlockQuantMatmulFusionPass.WEIGHT_NAME] ?: node.id
        context.emitComment("block_quant_matmul ${node.id}: $weightName ${storage.encoding.name} [$n, $k] as $codeType codes + bf16 scales")

        var x = operands[0]
        if (inElem != "bf16") {
            val v = context.nextTempValue()
            emit("$v = stablehlo.convert $x : (${tt(inShape, inElem)}) -> ${tt(inShape, "bf16")}")
            x = v
        }
        if (inShape != listOf(1, m, k)) {
            val v = context.nextTempValue()
            emit("$v = stablehlo.reshape $x : (${tt(inShape, "bf16")}) -> ${t(1, m, k)}")
            x = v
        }

        val codes = context.nextTempValue()
        emit("$codes = arith.constant dense<\"${hex(weight.codes)}\"> : ${t(nB, b, n, elem = codeType)}")
        val scales = context.nextTempValue()
        emit("$scales = arith.constant dense<\"${hex(weight.scales)}\"> : ${t(nB, 1, n)}")
        val zp = weight.zeroPoints?.let {
            val v = context.nextTempValue()
            emit("$v = arith.constant dense<\"${hex(it)}\"> : ${t(nB, 1, n)}")
            v
        }

        val init = context.nextTempValue()
        emit("$init = tensor.empty() : ${t(nB, b, n)}")
        val identity = "affine_map<(d0, d1, d2) -> (d0, d1, d2)>"
        val perBlock = "affine_map<(d0, d1, d2) -> (d0, 0, d2)>"
        val maps = if (zp == null) listOf(identity, perBlock, identity) else listOf(identity, perBlock, perBlock, identity)
        val ins = if (zp == null) "$codes, $scales : ${t(nB, b, n, elem = codeType)}, ${t(nB, 1, n)}"
        else "$codes, $scales, $zp : ${t(nB, b, n, elem = codeType)}, ${t(nB, 1, n)}, ${t(nB, 1, n)}"
        val (fl, df, r) = Triple(context.nextTempValue(), context.nextTempValue(), context.nextTempValue())
        val body = if (zp == null) {
            "^bb0(%c: $codeType, %s: bf16, %o: bf16):\n" +
                "      $fl = arith.sitofp %c : $codeType to bf16\n" +
                "      $r = arith.mulf $fl, %s : bf16\n"
        } else {
            "^bb0(%c: $codeType, %s: bf16, %z: bf16, %o: bf16):\n" +
                "      $fl = arith.sitofp %c : $codeType to bf16\n" +
                "      $df = arith.subf $fl, %z : bf16\n" +
                "      $r = arith.mulf $df, %s : bf16\n"
        }
        val dq = context.nextTempValue()
        emit(
            "$dq = linalg.generic {indexing_maps = [${maps.joinToString(", ")}], " +
                "iterator_types = [\"parallel\", \"parallel\", \"parallel\"]} ins($ins) outs($init : ${t(nB, b, n)}) {\n" +
                "    $body" +
                "      linalg.yield $r : bf16\n" +
                "    } -> ${t(nB, b, n)}"
        )
        val collapsed = context.nextTempValue()
        emit("$collapsed = tensor.collapse_shape $dq [[0, 1], [2]] : ${t(nB, b, n)} into ${t(k, n)}")
        val expanded = context.nextTempValue()
        emit("$expanded = tensor.expand_shape $collapsed [[0, 1], [2]] output_shape [1, $k, $n] : ${t(k, n)} into ${t(1, k, n)}")
        val zero = context.nextTempValue()
        emit("$zero = arith.constant 0.000000e+00 : bf16")
        val acc0 = context.nextTempValue()
        emit("$acc0 = tensor.empty() : ${t(1, m, n)}")
        val acc = context.nextTempValue()
        emit("$acc = linalg.fill ins($zero : bf16) outs($acc0 : ${t(1, m, n)}) -> ${t(1, m, n)}")
        var y = context.nextTempValue()
        emit("$y = linalg.batch_matmul ins($x, $expanded : ${t(1, m, k)}, ${t(1, k, n)}) outs($acc : ${t(1, m, n)}) -> ${t(1, m, n)}")

        if (outShape != listOf(1, m, n)) {
            val v = context.nextTempValue()
            emit("$v = stablehlo.reshape $y : (${t(1, m, n)}) -> ${tt(outShape, "bf16")}")
            y = v
        }
        if (outElem != "bf16") {
            val v = context.nextTempValue()
            emit("$v = stablehlo.convert $y : (${tt(outShape, "bf16")}) -> ${tt(outShape, outElem)}")
            y = v
        }
        return ConversionResult.Success(outputValueName = y, emittedOperations = ops)
    }

    private companion object {
        private val HEX = "0123456789ABCDEF".toCharArray()

        fun hex(bytes: ByteArray): String {
            val out = CharArray(2 + bytes.size * 2)
            out[0] = '0'; out[1] = 'x'
            for (i in bytes.indices) {
                val v = bytes[i].toInt() and 0xFF
                out[2 + 2 * i] = HEX[v ushr 4]
                out[3 + 2 * i] = HEX[v and 0xF]
            }
            return out.concatToString()
        }

        /** bf16 values as little-endian bytes, the order MLIR reads a hex dense attribute in. */
        fun hex(bf16: ShortArray): String {
            val bytes = ByteArray(bf16.size * 2)
            for (i in bf16.indices) {
                bytes[2 * i] = bf16[i].toByte()
                bytes[2 * i + 1] = (bf16[i].toInt() ushr 8).toByte()
            }
            return hex(bytes)
        }
    }
}

/**
 * A Q8_0 or Q4_0 weight `[N, K]` regrouped for the block-quant chain: `codes[b][j][o]` is element
 * `(o, 32b + j)`, `scales[b][o]` and `zeroPoints[b][o]` belong to block `b` of output row `o`.
 * The values it stands for are `(codes - zeroPoints) * scales`.
 */
internal class BlockQuantWeight(
    val k: Int,
    val n: Int,
    /** Code width: 8 for Q8_0, 4 for Q4_0 (one code per byte, sign-extended). */
    val bits: Int,
    /** `[nB, 32, N]` signed codes. */
    val codes: ByteArray,
    /** `[nB, N]` bf16 bits, all positive. */
    val scales: ShortArray,
    /** `[nB, N]` bf16 bits, 0 or -1; `null` when every block's scale was non-negative. */
    val zeroPoints: ShortArray?,
) {
    val blockCount: Int get() = k / BLOCK

    companion object {
        const val BLOCK: Int = 32
        private const val BF16_ONE: Short = 0x3F80
        private const val BF16_MINUS_ONE: Short = 0xBF80.toShort()

        fun of(storage: PackedBlockStorage): BlockQuantWeight {
            require(storage.shape.rank == 2) { "block-quant weight must be [out, in], got ${storage.shape.dimensions.toList()}" }
            val n = storage.shape[0]
            val k = storage.shape[1]
            require(k % BLOCK == 0) { "block-quant weight in = $k is not a multiple of $BLOCK" }
            val (bits, scaleOf, codeOf) = when (storage.encoding) {
                TensorEncoding.Q8_0 -> {
                    val q = storage as? Q8_0TensorData ?: throw IllegalArgumentException("Q8_0 storage ${storage::class.simpleName} has no block accessors")
                    Triple(8, q::getBlockScale, { blk: Int, j: Int -> q.getCode(blk, j).toInt() })
                }
                TensorEncoding.Q4_0 -> {
                    val q = storage as? Q4_0TensorData ?: throw IllegalArgumentException("Q4_0 storage ${storage::class.simpleName} has no block accessors")
                    Triple(4, q::getBlockScale, { blk: Int, j: Int -> q.getCode(blk, j).toInt() - 8 })
                }
                else -> throw IllegalArgumentException("block-quant matmul supports Q8_0 and Q4_0, not ${storage.encoding.name}")
            }
            val nB = k / BLOCK
            val codes = ByteArray(nB * BLOCK * n)
            val scales = ShortArray(nB * n)
            var zeroPoints: ShortArray? = null
            for (o in 0 until n) {
                for (b in 0 until nB) {
                    val blk = o * nB + b
                    val d = scaleOf(blk)
                    val at = b * n + o
                    when {
                        d > 0f -> {
                            scales[at] = bf16(d)
                            for (j in 0 until BLOCK) codes[(b * BLOCK + j) * n + o] = codeOf(blk, j).toByte()
                        }
                        d < 0f -> {
                            // (c - 0) * d == (-1 - c - (-1)) * -d, and -1 - c stays in range for i8 and i4.
                            scales[at] = bf16(-d)
                            val zp = zeroPoints ?: ShortArray(nB * n).also { zeroPoints = it }
                            zp[at] = BF16_MINUS_ONE
                            for (j in 0 until BLOCK) codes[(b * BLOCK + j) * n + o] = (-1 - codeOf(blk, j)).toByte()
                        }
                        // An all-zero block: zero codes, and a scale the kernel accepts (it rejects 0).
                        else -> scales[at] = BF16_ONE
                    }
                }
            }
            return BlockQuantWeight(k, n, bits, codes, scales, zeroPoints)
        }

        /** FP32 → bf16 bits, rounded to nearest even. */
        fun bf16(value: Float): Short {
            val bits = value.toRawBits()
            return ((bits + 0x7FFF + ((bits ushr 16) and 1)) ushr 16).toShort()
        }
    }
}
