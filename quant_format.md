GGUF K-Quant and I-Quant Format Details
K-Quantization Types (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K)

K-quants use a hierarchical block structure: weights are grouped into super-blocks (typically 256 values each) subdivided into smaller blocks. Each block has its own quantized scale (and offset for type-1) stored with reduced precision, and these block scale parameters are themselves quantized relative to a higher-level super-block scale. This structure yields better accuracy per bit than older methods. Below we detail each K-type quantization:

Q2_K (2-bit K-quantization, type-1 asymmetrical)

Block size & bytes: Uses super-blocks of 256 weights, divided into 16 blocks of 16 each
github.com
. Each super-block is stored in 84 bytes, which is ~2.6 bits per weight
github.com
huggingface.co
.

Scales/offsets: Each 16-weight block has an FP16 scale and offset that are shared across its 16 values. To save space, these are not stored directly but as 4-bit indices: for each block, a 4-bit scale index and 4-bit minimum (offset) index are packed into one byte
huggingface.co
huggingface.co
. The super-block contains two 16-bit (half-precision) values: d (super-block scale for the scales) and dmin (super-block scale for the offsets)
huggingface.co
. The 4-bit indices for each block are interpreted via these super-block factors (e.g. actual block scale = d * index_scale, and block offset = dmin * index_min). All FP16 values are stored in little-endian.

Code packing: Each weight is quantized to 2 bits (q in 0–3). These 2-bit codes are packed densely in an array qs. Four weights fit in one byte. For 256 weights per super-block, qs has 64 bytes
huggingface.co
. The 16 bytes of packed 4-bit scale/min indices precede the quant codes, followed by the 4 bytes of super-block FP16 scales, making up the 84-byte block
huggingface.co
.

Dequantization: To reconstruct a float, each weight’s 2-bit code q is scaled and shifted: w = q * scale_block + min_block, where scale_block = d * (scale_index/15) and min_block = dmin * (min_index/15) (the 4-bit indices are out of 15)
haroldbenoit.com
. In other words, the block’s 4-bit values define its scale and zero-point (offset) relative to the super-block’s FP16 scale factors. This is an asymmetric quantization scheme (type-1)
haroldbenoit.com
haroldbenoit.com
.

References & caveats: See ggml code block_q2_K for structure and size
huggingface.co
huggingface.co
. Ensure to handle the FP16 d and dmin with correct endianness. The 4-bit fields are packed one per nibble (high nibble vs low nibble corresponding to scale vs min for each block). When decoding, note that Q2_K in practice uses a mix with higher-bit for critical tensors (e.g. certain weight matrices may be stored in Q4_K to avoid excessive error
github.com
).

Q3_K (3-bit K-quantization, type-0 symmetrical)

Block size & bytes: Uses super-blocks of 256 weights, divided into 16 blocks of 16 each
github.com
. Each super-block is 110 bytes (3.4375 bits per weight)
github.com
huggingface.co
.

Scales: Each 16-value block shares a single scale. The block scales are quantized with 6-bit precision. All 16 block scale values (6 bits each) are packed into a 12-byte array in the super-block
huggingface.co
. Additionally, one FP16 super-block scale d is stored (2 bytes) to decode those 6-bit values
huggingface.co
. (No offset per block since this is symmetric quantization.)

Code packing: Each weight’s 3-bit quantized value is split across two arrays for efficient packing
huggingface.co
. The lower 2 bits of all 256 values are stored in a 64-byte array qs (each byte contains four 2-bit codes). The top bit of each value is stored in a separate 32-byte bitmask hmask (each byte contains 8 high-bits)
huggingface.co
. This yields 96 bytes for quant codes in total. The 6-bit scale data (12 bytes) and the 2-byte super-block scale follow, totaling 110 bytes
huggingface.co
.

Dequantization: Weights are reconstructed as w = q * scale_block. Here q (0–7) is formed by combining the bits from hmask and qs. The block’s actual scale is obtained by scale_block = d * (scale_index / 63) (using the 6-bit index from the scales array, 0–63 range). This is a symmetric scheme (offset is zero)
github.com
.

References & caveats: See block_q3_K in ggml code
huggingface.co
huggingface.co
. The packing of 3-bit codes requires combining the bitmask and low-bit array. Endianness applies to the FP16 d. The 6-bit scale indices are packed contiguously across 12 bytes (some bit operations needed to extract each 6-bit value). Overall, Q3_K has no per-block bias term, simplifying dequant math but using only scales
github.com
.

Q4_K (4-bit K-quantization, type-1 asymmetrical)

Block size & bytes: Uses super-blocks of 256 weights, divided into 8 blocks of 32 each
github.com
. Each super-block occupies 144 bytes total (4.5 bits per weight)
github.com
huggingface.co
.

Scales/offsets: Every 32-weight block has its own scale and bias (min). These are quantized to 6 bits each. The super-block stores an array of 12 bytes (scales) that encodes all 8 blocks’ 6-bit scale and 6-bit min values
github.com
github.com
. (They are packed in a specific bit-interleaving pattern across the 12 bytes for SIMD friendliness
github.com
.) A pair of FP16 values (d and dmin) are stored per super-block to decode those 6-bit fields
huggingface.co
huggingface.co
.

Code packing: Each weight’s 4-bit quantized value is stored in the 128-byte qs array (since 4 bits per weight, 2 values per byte)
huggingface.co
. No separate high-bit mask is needed for 4-bit values. The layout in memory is: first the 4 bytes of super-block FP16 scales, then 12 bytes of packed scale/min indices, then 128 bytes of quant codes.

Dequantization: w = q * scale_block + min_block. Here q∈[0,15] is the 4-bit weight code. The block’s scale and offset are recovered by: scale_block = d * (scale_index/63) and min_block = dmin * (min_index/63), using the 6-bit indices (0–63 range) from the scales array
huggingface.co
huggingface.co
. This provides a linear scaling and bias per block (asymmetric quantization).

References & caveats: See block_q4_K definition
huggingface.co
huggingface.co
. The 6-bit fields are packed non-trivially; the ggml implementation uses bit-masks/shifts to unpack them
github.com
github.com
. Ensure FP16 values are read with correct endianness. Q4_K’s 4.5 bits/weight is the same storage cost as legacy Q4_0, but it offers better accuracy by using per-block offsets
huggingface.co
huggingface.co
.

Q5_K (5-bit K-quantization, type-1 asymmetrical)

Block size & bytes: Uses super-blocks of 256 weights, divided into 8 blocks of 32 each (same block structure as Q4_K)
github.com
. Each super-block is 176 bytes (5.5 bits per weight)
github.com
huggingface.co
.

Scales/offsets: Each 32-value block has a scale and offset quantized to 6 bits each (like Q4_K). The super-block stores these in a packed array of 12 bytes (holding 8×(6+6) bits). It also stores FP16 d and dmin values per super-block for decoding the 6-bit fields
huggingface.co
huggingface.co
.

Code packing: 5-bit weight codes are split across two arrays for efficiency. The lower 4 bits of each of the 256 values are in a 128-byte array qs. The top bit of each value (the 5th bit) is in a separate 32-byte array qh (each byte of qh packs 8 high-bits)
huggingface.co
. Thus, qs+qh represent all 5-bit codes (160 bytes). Along with 12 bytes of scale data and 4 bytes for d/dmin, that totals 176 bytes
huggingface.co
.

Dequantization: w = q * scale_block + min_block, with q∈[0,31]. The 5-bit code is reconstructed from qs and qh. The block’s actual scale = d * (scale_index/63) and offset = dmin * (min_index/63) as given by the 6-bit indices (same interpretation as Q4_K)
huggingface.co
huggingface.co
.

References & caveats: See block_q5_K in code
huggingface.co
huggingface.co
. Note the use of two arrays for quant data: you must combine qh and qs bits to get the 5-bit value for each weight. Endianness considerations apply to FP16 super-block values. Q5_K yields about 5.5 bits/weight, significantly more compact than older 5-bit schemes (which required ~6 bits/weight)
huggingface.co
huggingface.co
.

Q6_K (6-bit K-quantization, type-0 symmetrical)

Block size & bytes: Uses super-blocks of 256 weights, divided into 16 blocks of 16 each
github.com
. Each super-block is 210 bytes (6.5625 bits per weight)
github.com
huggingface.co
.

Scales: Every 16-weight block has a single scale (no block offset, since symmetric). Block scales are stored with 8-bit precision. The super-block holds 16 scale values (one per block) in a 16-byte array (type int8_t)
huggingface.co
. One FP16 super-block scale d is included (2 bytes) to convert those 8-bit scale values to real scales
huggingface.co
.

Code packing: Each weight’s 6-bit quant code is split into two parts. The lower 4 bits of all values are in a 128-byte array ql. The upper 2 bits of each value are in a 64-byte array qh (each byte packs four 2-bit pairs)
huggingface.co
. Together they represent all 6-bit codes (192 bytes). Adding 16 bytes of scales and 2 bytes of d gives 210 bytes total
huggingface.co
.

Dequantization: w = q * scale_block. The 6-bit q (0–63) is reconstructed from ql and qh. The block’s real scale is scale_block = d * (scale_index/127) if we treat the 8-bit scale index (0–255 in int8) as mapping into 0–127 range for positive scales (the ggml implementation may use the signed 8-bit directly as well)
huggingface.co
huggingface.co
. No offset is added (type-0 symmetric). Essentially each block’s max magnitude is captured by the 8-bit scale index.

References & caveats: See block_q6_K in code
huggingface.co
huggingface.co
. Q6_K uses only one FP16 super-scale (no dmin). The struct is padded to 210 bytes; no extra alignment beyond that
huggingface.co
. Be mindful that the int8_t scale array may contain negative values – in practice ggml treats them as unsigned 8-bit for scale magnitude (the sign bit might be unused or reserved). Q6_K models are larger but often higher fidelity; on CPU they are ~44% larger than Q4_K in memory, which can affect speed
github.com
github.com
.

Q8_K (8-bit K-quantization, symmetric, for intermediates)

Block size & bytes: Uses blocks of 256 weights (no further subdivision)
github.com
. Each block is stored in 292 bytes. Q8_K is typically used for intermediate results or partial quantization; it’s not a common choice for full model weights due to its size (it’s essentially 8-bit)
github.com
.

Scales: Each 256-length block has a single scale factor d stored as a 32-bit float (4 bytes)
huggingface.co
. (Float is used here instead of FP16 to avoid precision loss in intermediate math.)

Code storage: Every weight’s quantized value is an 8-bit integer stored in a 256-byte array qs (one int8 per weight)
huggingface.co
. In addition, Q8_K stores a 32-byte array bsums, containing 16-bit integers that are the sums of each 16 consecutive quant values (16 groups of 16)
huggingface.co
. These precomputed sums accelerate dot-product calculations in ggml.

Dequantization: w = q * d. Since each weight’s code q is 8-bit and d is a float scale, reconstruction is simply a multiplication (no offset)
huggingface.co
. The bsums are not needed for dequantization itself, but rather as an optimization for matrix multiplication (you might recompute or ignore them depending on your implementation).

References & caveats: See block_q8_K structure
huggingface.co
. Note that Q8_K appears as “Q8_K (block size 256)” in GGUF and differs from legacy Q8_0 primarily by using a larger block and providing those bsums for speed. Ensure you handle the float scale properly (little-endian 32-bit). Since Q8_K is usually used internally (e.g. mixed with lower-bit blocks in some mixed-quant schemes), its presence in final model files is less common.

I-Quantization Types (IQ2, IQ3, IQ4, etc.)

I-quants (importance-based quantization) build on K-quants by using an importance matrix during quantization to allocate precision where needed most. They still use 256-weight super-blocks but achieve better quality-to-size ratios, especially at very low bit widths, by non-uniform bit allocation or non-linear scaling of weights based on their estimated importance
stackoverflow.com
gist.github.com
. The exact decoding formulas are more complex (often involving second-order calibration or look-up tables) rather than a simple linear q*d + m. Below are documented IQ formats:

IQ4_NL (4-bit “Non-Linear” I-Quant): 4-bit weights with a non-linear quantization mapping
github.com
. Super-block of 256 weights, organized as 16 blocks of 16 for internal computation
github.com
. Achieves about 4.5 bits per weight
huggingface.co
. Instead of a simple linear scale, it applies a non-linear transform to the quantized values (as indicated by “NL”). This yields higher accuracy by better handling outlier weights. References: GGML PR introducing IQ4_NL
github.com
. (Internally, it still stores an FP16 super-block scale and uses an importance matrix to adjust weights, but the scale application is non-linear.)

IQ4_XS (4-bit I-Quant, eXtra Small): A 4-bit importance-quantized format with about 4.25 bits per weight
huggingface.co
. Super-block of 256, likely split into 8 blocks of 32 for bit allocation
github.com
. Uses a linear reconstruction with importance-weighted scaling (hence requires an importance matrix). This format offers a smaller model size than IQ4_NL at some accuracy cost. References: Official docs list IQ4_XS
huggingface.co
.

IQ3_S (3-bit I-Quant, Small): 3-bit weights with importance-based scaling. ~3.44 bits per weight
huggingface.co
. Super-block of 256 weights (internally likely 16×16 blocks)
github.com
. It uses a linear formula w.r.t. a super-block scale and importance metrics, improving on plain Q3_K. References: Listed in GGUF spec
huggingface.co
 (introduced as a higher-accuracy alternative to Q3_K
huggingface.co
).

IQ3_XXS (3-bit I-Quant, eXtra Extra Small): An even more compressed 3-bit format (~3.06 bits/weight)
huggingface.co
. Likely uses 8 blocks of 32 within the 256 super-block
github.com
 to minimize overhead. Yields smaller size at some accuracy trade-off compared to IQ3_S. Relies on importance calibration to preserve model quality. References: GGUF spec table
huggingface.co
.

IQ2_XXS (2-bit I-Quant, eXtra Extra Small): An extremely compressed 2-bit format at ~2.06 bits/weight
huggingface.co
. It effectively stores 256 weights with just one FP16 scale for the whole super-block (no sub-block splits)
raw.githubusercontent.com
github.com
. This is essentially a “true” 2-bit quantization with only a global scale (the 0.06 bits overhead comes from that FP16 scale
huggingface.co
). Requires a robust importance matrix to maintain fidelity at such low precision.

IQ2_XS (2-bit I-Quant, eXtra Small): ~2.31 bits/weight
huggingface.co
. Likely uses a moderate overhead scheme (e.g. 16 blocks of 16) to allocate a few extra bits for scaling. This provides better accuracy than IQ2_XXS while still being extremely compact. References: Spec entry
huggingface.co
.

IQ2_S (2-bit I-Quant, Small): ~2.50 bits/weight
huggingface.co
. This format uses additional scaling degrees of freedom (higher overhead than XS) – for example, subdividing the super-block further (possibly 32 blocks of 8) to give more per-block scales. It strikes a balance between size and accuracy, sitting between Q2_K (2.62 bpw) and the more aggressive IQ2_XS. References: Spec entry
huggingface.co
.

IQ1_S (1-bit I-Quant, Small): 1-bit weights with importance-driven scaling, ~1.56 bits/weight
huggingface.co
. Even though weights are stored as 0/1 (1-bit), extra scaling info brings it to ~1.5 bits per weight. Internally, this uses 8 blocks of 32 within 256 (each block 1-bit quantized, plus scale)
github.com
. An FP16 super-scale is applied along with importance weighting to reconstruct values. This format is extremely lossy but the importance matrix helps retain key information.

IQ1_M (1-bit I-Quant, Medium): 1-bit weights with a bit more overhead (~1.75 bits/weight)
huggingface.co
. Uses 16 blocks of 16 (more block-wise scales) for the 256-weight super-block
github.com
. This gives each group of 16 weights its own scale (quantized), improving accuracy over IQ1_S at the cost of a slightly larger model. Still an heavily quantized format reserved for scenarios where maximum compression is needed. References: Spec entry and PR
huggingface.co
github.com
.

Note: For all IQ formats, the general storage pattern is similar to K-quants: a super-block FP16 scale and a set of block-wise quantized scale factors (and possibly offsets) are stored, along with the quantized weight codes. The key difference is that the mapping from quant code to actual weight is not purely linear per block. It involves the importance matrix calibration data, which effectively modulates the scales based on weight sensitivity
medium.com
stackoverflow.com
. In implementation, this means custom dequantization logic per format (often found in the dequantize_row_xx functions of GGML). Endianness is still little-endian for all multi-byte values (FP16/FP32 scales, etc.). Also note that many IQ formats were introduced recently and might not be supported on all backends (for example, some GPU kernels might lack IQ4_XS support as of writing
github.com
). When implementing, use the GGML/llama.cpp reference code as a guide for the exact bit-packing and reconstruction algorithm for each IQ variant.

Ternary Quantization Types (TQ1_0, TQ2_0) - BitNet Support

Ternary quantization formats store weights using only three values: {-1, 0, +1}. These are designed for BitNet-style models where weights are constrained to ternary values during training. The key advantage is that matrix multiplication becomes addition-only (no floating-point multiplies), enabling significant speedups on hardware that supports it.

Both TQ formats encode ternary values as {0, 1, 2} which map to {-1, 0, +1} respectively. Dequantization formula: `output[i] = (encoded_value - 1) * scale`

TQ2_0 (Ternary 2-bit, Simple)

Block size & bytes: Uses blocks of 256 weights. Each block is 66 bytes (~2.06 bits per weight).

Layout:
- 64 bytes: Quantized data (4 ternary values per byte, 2-bit each)
- 2 bytes: FP16 scale factor

Code packing: Each byte stores 4 ternary values using 2 bits each:
- bits 0-1: value 0 (v0)
- bits 2-3: value 1 (v1)
- bits 4-5: value 2 (v2)
- bits 6-7: value 3 (v3)

Dequantization: For each 2-bit encoded value e (0, 1, or 2):
```
w = (e - 1) * scale
```
Where scale is the FP16 value from bytes 64-65 (little-endian).

References: Added in llama.cpp PR #8151. GGUF type value: 35.

TQ1_0 (Ternary Base-3, Compact)

Block size & bytes: Uses blocks of 256 weights. Each block is 54 bytes (~1.69 bits per weight).

Layout:
- 48 bytes: Base-3 packed data (5 ternary values per byte, 240 elements total)
- 4 bytes: 2-bit packed data for remaining 16 elements
- 2 bytes: FP16 scale factor

Code packing (base-3 region): Each byte encodes 5 ternary values using base-3 arithmetic:
```
byte_value = v0 + v1*3 + v2*9 + v3*27 + v4*81
```
Where each v ∈ {0, 1, 2}. Since 3^5 = 243 < 256, this fits in a single byte.

To decode, extract values using modulo/division:
```
v0 = byte_value % 3
v1 = (byte_value / 3) % 3
v2 = (byte_value / 9) % 3
v3 = (byte_value / 27) % 3
v4 = (byte_value / 81) % 3
```

Code packing (2-bit region): The last 16 values (indices 240-255) use standard 2-bit packing like TQ2_0.

Dequantization: Same as TQ2_0:
```
w = (encoded_value - 1) * scale
```

TQ1_0 achieves better compression (~1.69 bpw) than TQ2_0 (~2.06 bpw) by using base-3 encoding for most values. The 16 trailing 2-bit values are needed because 48 bytes × 5 values = 240, leaving 16 values for the 256-element block.

References: Added in llama.cpp PR #8151. GGUF type value: 34.

Native Ternary Inference

For maximum performance with ternary weights, instead of dequantizing to FP32, specialized kernels can perform "ternary matmul" directly:
```
output[i] = sum over j of: activation[j] * ternary_weight[j,i]
          = sum over j where weight=+1 of: activation[j]
          - sum over j where weight=-1 of: activation[j]
```

This replaces all multiplications with additions and subtractions, which is significantly faster on many hardware platforms. SKaiNET's BitNet support includes native ternary kernels in addition to dequantization for compatibility.

Sources: Official GGUF quantization documentation and code in ggml/llama.cpp
huggingface.co
github.com
huggingface.co
huggingface.co
, as well as discussions of K-quant and I-quant implementations in the GGML community
github.com
stackoverflow.com
. These references and the static assertions in code confirm block sizes, packing layouts, and formulas. Always ensure to handle byte-order and alignment exactly as specified by the GGUF format to correctly reconstruct the original FP32 values.
