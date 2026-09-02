package sk.ainet.io.safetensors

import sk.ainet.context.ExecutionContext
import sk.ainet.io.ParametersLoader
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.DTypePolicy
import kotlin.reflect.KClass

/**
 * ParametersLoader implementation for SafeTensors format using streaming.
 *
 * Uses [StreamingSafeTensorsReader] for memory-efficient loading - only
 * parses the header (~1KB-1MB) and loads tensors on-demand.
 *
 * Supported conversions (see [SafeTensorsMaterializer]):
 * - F32/F64 tensors -> FP32 (F64 downcast with warning)
 * - I32/I64 tensors -> Int32 (I64 downcast with warning)
 * - I8/U8 tensors -> Int8
 * - F16 tensors -> FP32 (with dequantization)
 * - BF16 tensors -> FP32 (default) OR native BF16 storage (`bf16Policy = KEEP_NATIVE`)
 *
 * Where possible, decoded arrays are wrapped (borrowed) rather than copied
 * into TensorData, avoiding a second allocation. The raw-byte decode step
 * (little-endian bytes → typed array) is still necessary.
 *
 * Single-file only: for HF checkpoints shipped as `model.safetensors.index.json`
 * plus `model-NNNNN-of-NNNNN.safetensors` shards, use
 * [ShardedSafeTensorsParametersLoader] (#1246).
 *
 * @param sourceProvider Factory providing RandomAccessSource to the SafeTensors file
 * @param onProgress Optional progress callback (current, total, tensorName)
 * @param bf16Policy How to handle `BFLOAT16` tensors. Default is
 *   [Bf16LoadPolicy.DEQUANT_TO_FP32] — backward-compatible with all
 *   existing consumers. Flip to [Bf16LoadPolicy.KEEP_NATIVE] to keep
 *   weights in their on-disk BF16 layout and let the matmul dispatch
 *   route to a vectorised BF16 kernel.
 */
class SafeTensorsParametersLoader(
    private val sourceProvider: () -> RandomAccessSource,
    private val onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> },
    private val bf16Policy: Bf16LoadPolicy = NarrowFloatLoadPolicy.DEQUANT_TO_FP32,
    private val fp16Policy: NarrowFloatLoadPolicy = NarrowFloatLoadPolicy.DEQUANT_TO_FP32,
) : ParametersLoader {

    override suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ) {
        StreamingSafeTensorsReader.open(sourceProvider()).use { reader ->
            val tensors = reader.tensors
            val total = tensors.size.toLong()
            var current = 0L

            for (tensorInfo in tensors) {
                val bytes = reader.loadTensorData(tensorInfo)
                val shape = Shape(*tensorInfo.shape.map { it.toInt() }.toIntArray())

                val tensor: Tensor<T, V> = SafeTensorsMaterializer.materialize(
                    ctx = ctx,
                    dtype = dtype,
                    name = tensorInfo.name,
                    dataType = tensorInfo.dataType,
                    rawDtype = tensorInfo.dtype,
                    shape = shape,
                    bytes = bytes,
                    bf16Policy = bf16Policy,
                    fp16Policy = fp16Policy,
                )

                onTensorLoaded(tensorInfo.name, tensor)
                current++
                onProgress(current, total, tensorInfo.name)
            }
        }
    }

    companion object {

        /**
         * Constructs a SafeTensorsParametersLoader from a generalised
         * [DTypePolicy] instead of the BF16-specific [Bf16LoadPolicy].
         * Bridge for the policy-driven loader path described in the
         * dtype-policy RFC (#615).
         *
         * Policy → behaviour mapping (BF16 source tensors only —
         * other dtypes are handled per the per-arm `require` checks
         * in [load]):
         * - [DTypePolicy.Any]: BF16 dequants to FP32 (the existing
         *   adaptive default).
         * - [DTypePolicy.Require] target = `BF16`: KEEP_NATIVE.
         * - [DTypePolicy.Require] target = `FP32`: DEQUANT_TO_FP32.
         * - [DTypePolicy.Require] target = `FP16`: throws — F16
         *   KEEP_NATIVE is a follow-up (no `Fp16DenseTensorData`
         *   yet); use `Require(FP32)` if you want F16 dequanted, or
         *   `Any` to inherit the adaptive default.
         * - [DTypePolicy.Require] target = anything else: throws —
         *   SafeTensors can't fabricate dtypes the file doesn't carry.
         * - [DTypePolicy.Prefer] target = `BF16`: KEEP_NATIVE.
         * - [DTypePolicy.Prefer] target = anything else: DEQUANT_TO_FP32
         *   (the soft path falls through).
         * - [DTypePolicy.OneOf] containing `BF16`: KEEP_NATIVE.
         * - [DTypePolicy.OneOf] without `BF16`: DEQUANT_TO_FP32.
         */
        fun withPolicy(
            sourceProvider: () -> RandomAccessSource,
            policy: DTypePolicy,
            onProgress: (current: Long, total: Long, message: String?) -> Unit = { _, _, _ -> },
        ): SafeTensorsParametersLoader = SafeTensorsParametersLoader(
            sourceProvider = sourceProvider,
            onProgress = onProgress,
            bf16Policy = mapPolicyToBf16(policy),
            fp16Policy = mapPolicyToFp16(policy),
        )

        internal fun mapPolicyToBf16(policy: DTypePolicy): Bf16LoadPolicy =
            SafeTensorsMaterializer.mapPolicyToBf16(policy)

        internal fun mapPolicyToFp16(policy: DTypePolicy): NarrowFloatLoadPolicy =
            SafeTensorsMaterializer.mapPolicyToFp16(policy)
    }
}
