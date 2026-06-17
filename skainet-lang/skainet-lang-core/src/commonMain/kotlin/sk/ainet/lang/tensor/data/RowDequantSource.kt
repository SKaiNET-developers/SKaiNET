package sk.ainet.lang.tensor.data

/**
 * Marker for a 2-D [TensorData] whose rows can be **dequantised on demand**, for tables that cannot (or
 * should not) be materialised as a single dense `FloatArray` — e.g. a packed-quant embedding whose logical
 * size exceeds `Int.MAX_VALUE` elements / 2 GB, or one kept packed to save memory.
 *
 * Such a tensor declares its **logical** dtype `FP32` (the dequantised value type); its packed bytes are an
 * internal storage detail, and `get`/`copyToFloatArray()` are typically unsupported. Ops that read whole
 * rows — primarily **embedding lookup** (`ops.gather` / `ops.indexSelect`, `dim = 0`, indices = token ids)
 * — MUST use [dequantRow] instead of element access, dequantising only the rows actually touched.
 *
 * This is the engine-level home of the contract; model-specific implementations (e.g. a GGUF Q6_K /
 * SafeTensors BF16 embedding) provide [dequantRow] over their own packed source.
 */
public interface RowDequantSource {
    /** Dequantise logical row [rowIdx] (`0 until shape[0]`) to a fresh `FloatArray` of length `shape[1]`. */
    public fun dequantRow(rowIdx: Int): FloatArray
}
