package sk.ainet.io.model

/**
 * Which way round a loaded 2-D weight's shape is (#973 census contradiction #6; #1098).
 *
 * GGUF stores dimensions in `ne` order — fastest-varying first — so a weight the rest of the world
 * calls `[out, in]` is written as `ne = [in, out]`. The bytes are the same either way: row-major
 * with the input dimension fastest, which *is* `[out, in]` row-major. Only the label differs.
 *
 * That label matters, because everything downstream of the loader assumes `[out, in]`: the block
 * grid of a packed weight is `out × blocksPerRow`, so a relayout driven by an `[in, out]` shape
 * computes the wrong permutation — or refuses, when `out` is not a multiple of the block size. Both
 * failures are in the census.
 */
public enum class WeightOrientation {
    /**
     * The file's own order, unreversed — GGUF `ne`, so `[in, out]` for a 2-D weight. What the
     * streaming loader has always produced, and the default, because changing it changes every
     * consumer's idea of a tensor's shape.
     */
    AS_STORED,

    /**
     * Logical `[out, in]`: the convention the engine, HF checkpoints and every kernel assume, and
     * the one the block relayout needs. Reverses a 2-D weight's dimensions at the load boundary;
     * the bytes are untouched, because they already are `[out, in]` row-major.
     */
    OUT_IN,
}
