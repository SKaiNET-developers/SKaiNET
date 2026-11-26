// Reference data for the GGUF IQ1/IQ2/IQ3/IQ4 and MXFP4 dequantizers (SKaiNET #1409).
//
// Links against llama.cpp's own ggml-quants.c and prints, per type, a few cases of
// raw block bytes together with what llama.cpp's dequantize_row_* returns for them.
// It also prints the constant tables the decoders index (grids, sign table, value table).
// Output: JSON on stdout. Build and run through gguf_iq_reference.py.
#define GGML_COMMON_IMPL_C      // this file gets its own copy of the constant tables, to print them
#include "ggml.h"
#include "ggml-common.h"
#include "ggml-quants.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- the four ggml.c symbols ggml-quants.c needs --------------------------------------------

typedef struct { enum ggml_type type; const char * name; size_t block_bytes; int block_elems; } type_info;

static const type_info TYPES[] = {
    { GGML_TYPE_MXFP4,   "MXFP4",   sizeof(block_mxfp4),   QK_MXFP4 },
    { GGML_TYPE_IQ2_XXS, "IQ2_XXS", sizeof(block_iq2_xxs), QK_K },
    { GGML_TYPE_IQ2_XS,  "IQ2_XS",  sizeof(block_iq2_xs),  QK_K },
    { GGML_TYPE_IQ2_S,   "IQ2_S",   sizeof(block_iq2_s),   QK_K },
    { GGML_TYPE_IQ3_XXS, "IQ3_XXS", sizeof(block_iq3_xxs), QK_K },
    { GGML_TYPE_IQ3_S,   "IQ3_S",   sizeof(block_iq3_s),   QK_K },
    { GGML_TYPE_IQ1_S,   "IQ1_S",   sizeof(block_iq1_s),   QK_K },
    { GGML_TYPE_IQ1_M,   "IQ1_M",   sizeof(block_iq1_m),   QK_K },
    // Already implemented in SKaiNET; included so the same fixture checks the existing decoders.
    { GGML_TYPE_IQ4_NL,  "IQ4_NL",  sizeof(block_iq4_nl),  QK4_NL },
    { GGML_TYPE_IQ4_XS,  "IQ4_XS",  sizeof(block_iq4_xs),  QK_K },
};
#define N_TYPES ((int)(sizeof(TYPES) / sizeof(TYPES[0])))

static const type_info * info_of(enum ggml_type t) {
    for (int i = 0; i < N_TYPES; ++i) if (TYPES[i].type == t) return &TYPES[i];
    fprintf(stderr, "harness: no size known for ggml type %d\n", (int) t);
    exit(2);
}

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "%s:%d: ", file, line); vfprintf(stderr, fmt, ap); fputc('\n', stderr);
    va_end(ap);
    exit(3);
}
size_t ggml_type_size(enum ggml_type t) { return info_of(t)->block_bytes; }
const char * ggml_type_name(enum ggml_type t) { return info_of(t)->name; }
size_t ggml_row_size(enum ggml_type t, int64_t ne) {
    const type_info * i = info_of(t);
    return i->block_bytes * (size_t) ne / (size_t) i->block_elems;
}

// What ggml_quantize_init does in ggml.c, which is not linked here.
static void ggml_quantize_init_for(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:   case GGML_TYPE_IQ1_M:   iq2xs_init_impl(type); break;
        case GGML_TYPE_IQ3_XXS: iq3xs_init_impl(256); break;
        case GGML_TYPE_IQ3_S:   iq3xs_init_impl(512); break;
        default: break;
    }
}

// ---- deterministic input ----------------------------------------------------------------------

static uint64_t rng_state;
static uint32_t rng_next(void) {            // xorshift64*, same sequence on every platform
    rng_state ^= rng_state >> 12; rng_state ^= rng_state << 25; rng_state ^= rng_state >> 27;
    return (uint32_t) ((rng_state * 0x2545F4914F6CDD1DULL) >> 32);
}

// A positive, normal half-precision scale between 2^-9 and 2^-3 with a random mantissa.
static uint16_t sane_half(void) {
    const uint32_t r = rng_next();
    const uint16_t exp = (uint16_t) (6 + (r % 7));      // biased exponent 6..12 -> 2^-9 .. 2^-3
    return (uint16_t) ((exp << 10) | ((r >> 8) & 0x3ff));
}

static void put_u16(uint8_t * p, uint16_t v) { p[0] = (uint8_t) (v & 0xff); p[1] = (uint8_t) (v >> 8); }

// Give block `b` of `type` a finite scale; everything else in the block stays as it is.
static void set_scale(enum ggml_type type, uint8_t * b, uint16_t half) {
    switch (type) {
        case GGML_TYPE_MXFP4:
            b[0] = (uint8_t) (112 + half % 24);          // E8M0 exponent: scale 2^-16 .. 2^7
            break;
        case GGML_TYPE_IQ1_M: {
            // The fp16 scale is spread over the top nibble of the four little-endian uint16 `scales`.
            uint8_t * sc = b + QK_K/8 + QK_K/16;
            for (int i = 0; i < 4; ++i) {
                const uint8_t nib = (uint8_t) ((half >> (4*i)) & 0xf);
                sc[2*i + 1] = (uint8_t) ((sc[2*i + 1] & 0x0f) | (nib << 4));
            }
            break;
        }
        default:
            put_u16(b, half);                             // ggml_half d, first field, little-endian
            break;
    }
}

// ---- output -----------------------------------------------------------------------------------

static void print_hex(const uint8_t * p, size_t n) {
    for (size_t i = 0; i < n; ++i) printf("%02x", p[i]);
}

static void dequantize(enum ggml_type type, const uint8_t * blocks, float * y, int64_t n) {
    switch (type) {
        case GGML_TYPE_MXFP4:   dequantize_row_mxfp4  ((const block_mxfp4 *)   blocks, y, n); break;
        case GGML_TYPE_IQ2_XXS: dequantize_row_iq2_xxs((const block_iq2_xxs *) blocks, y, n); break;
        case GGML_TYPE_IQ2_XS:  dequantize_row_iq2_xs ((const block_iq2_xs *)  blocks, y, n); break;
        case GGML_TYPE_IQ2_S:   dequantize_row_iq2_s  ((const block_iq2_s *)   blocks, y, n); break;
        case GGML_TYPE_IQ3_XXS: dequantize_row_iq3_xxs((const block_iq3_xxs *) blocks, y, n); break;
        case GGML_TYPE_IQ3_S:   dequantize_row_iq3_s  ((const block_iq3_s *)   blocks, y, n); break;
        case GGML_TYPE_IQ1_S:   dequantize_row_iq1_s  ((const block_iq1_s *)   blocks, y, n); break;
        case GGML_TYPE_IQ1_M:   dequantize_row_iq1_m  ((const block_iq1_m *)   blocks, y, n); break;
        case GGML_TYPE_IQ4_NL:  dequantize_row_iq4_nl ((const block_iq4_nl *)  blocks, y, n); break;
        case GGML_TYPE_IQ4_XS:  dequantize_row_iq4_xs ((const block_iq4_xs *)  blocks, y, n); break;
        default: exit(2);
    }
}

static size_t quantize(enum ggml_type type, const float * src, uint8_t * dst, int64_t n, const float * imatrix) {
    switch (type) {
        case GGML_TYPE_MXFP4:   return quantize_mxfp4  (src, dst, 1, n, imatrix);
        case GGML_TYPE_IQ2_XXS: return quantize_iq2_xxs(src, dst, 1, n, imatrix);
        case GGML_TYPE_IQ2_XS:  return quantize_iq2_xs (src, dst, 1, n, imatrix);
        case GGML_TYPE_IQ2_S:   return quantize_iq2_s  (src, dst, 1, n, imatrix);
        case GGML_TYPE_IQ3_XXS: return quantize_iq3_xxs(src, dst, 1, n, imatrix);
        case GGML_TYPE_IQ3_S:   return quantize_iq3_s  (src, dst, 1, n, imatrix);
        case GGML_TYPE_IQ1_S:   return quantize_iq1_s  (src, dst, 1, n, imatrix);
        case GGML_TYPE_IQ1_M:   return quantize_iq1_m  (src, dst, 1, n, imatrix);
        case GGML_TYPE_IQ4_NL:  return quantize_iq4_nl (src, dst, 1, n, imatrix);
        case GGML_TYPE_IQ4_XS:  return quantize_iq4_xs (src, dst, 1, n, imatrix);
        default: exit(2);
    }
}

static int first_case;
static void emit_case(const type_info * t, const char * name, const char * what, const uint8_t * blocks, int nblocks) {
    const int64_t n = (int64_t) nblocks * t->block_elems;
    float * y = malloc((size_t) n * sizeof(float));
    dequantize(t->type, blocks, y, n);
    printf("%s\n      {\"name\": \"%s\", \"description\": \"%s\", \"blocks\": %d,\n       \"bytes\": \"", first_case ? "" : ",", name, what, nblocks);
    print_hex(blocks, (size_t) nblocks * t->block_bytes);
    printf("\",\n       \"f32le\": \"");
    print_hex((const uint8_t *) y, (size_t) n * sizeof(float));   // little-endian host assumed (checked in main)
    printf("\"}");
    first_case = 0;
    free(y);
}

int main(void) {
    const uint16_t endian_probe = 1;
    if (*(const uint8_t *) &endian_probe != 1) { fprintf(stderr, "harness: little-endian host required\n"); return 2; }

    // The tables as the bytes a little-endian C program reads through `(const uint8_t *)(table + i)`.
    printf("{\n  \"tables\": [");
#define EMIT_TABLE(first, tbl, signedness) \
    printf("%s\n    {\"name\": \"%s\", \"entries\": %zu, \"entry_bytes\": %zu, \"element\": \"%s\", \"bytes\": \"", first ? "" : ",", #tbl, \
           sizeof(tbl) / sizeof(tbl[0]), sizeof(tbl[0]), signedness); \
    print_hex((const uint8_t *) tbl, sizeof(tbl)); printf("\"}")
    EMIT_TABLE(1, kmask_iq2xs,  "uint8");
    EMIT_TABLE(0, ksigns_iq2xs, "uint8");
    EMIT_TABLE(0, iq2xxs_grid,  "uint8");
    EMIT_TABLE(0, iq2xs_grid,   "uint8");
    EMIT_TABLE(0, iq2s_grid,    "uint8");
    EMIT_TABLE(0, iq3xxs_grid,  "uint8");
    EMIT_TABLE(0, iq3s_grid,    "uint8");
    EMIT_TABLE(0, iq1s_grid,    "int8");
    EMIT_TABLE(0, kvalues_mxfp4, "int8");
    EMIT_TABLE(0, kvalues_iq4nl, "int8");
    printf("\n  ],\n  \"iq1_delta\": %.9g,\n  \"types\": [", (double) IQ1S_DELTA);
    for (int ti = 0; ti < N_TYPES; ++ti) {
        const type_info * t = &TYPES[ti];
        const size_t bb = t->block_bytes;
        rng_state = 0x9E3779B97F4A7C15ULL ^ ((uint64_t) t->type * 0x100000001B3ULL);

        printf("%s\n    {\"name\": \"%s\", \"ggml_type\": %d, \"block_bytes\": %zu, \"block_elements\": %d, \"cases\": [",
               ti ? "," : "", t->name, (int) t->type, bb, t->block_elems);
        first_case = 1;

        // 1. Random bytes: every index, sign and sub-scale field takes arbitrary values. Only the
        //    block scale is replaced by a finite one, so no NaN appears in the expected values.
        {
            enum { NB = 4 };
            uint8_t * blocks = malloc(NB * bb);
            for (size_t i = 0; i < NB * bb; ++i) blocks[i] = (uint8_t) (rng_next() >> 24);
            for (int b = 0; b < NB; ++b) set_scale(t->type, blocks + b * bb, sane_half());
            emit_case(t, "random", "random block bytes with a finite block scale", blocks, NB);
            free(blocks);
        }

        // 2. All payload bits clear and all payload bits set, with the same finite scale.
        {
            uint8_t * blocks = malloc(2 * bb);
            memset(blocks, 0x00, bb);
            memset(blocks + bb, 0xff, bb);
            set_scale(t->type, blocks, 0x2c00);        // 2^-4 as fp16
            set_scale(t->type, blocks + bb, 0x2c00);
            emit_case(t, "extremes", "one block of 0x00 and one of 0xff, block scale 2^-4 (MXFP4: exponent byte 120)", blocks, 2);
            free(blocks);
        }

        // 3. What llama.cpp's own quantizer writes for a fixed pseudo-random signal.
        {
            enum { N = 512 };
            float src[N], imatrix[N];
            for (int i = 0; i < N; ++i) {
                float s = 0.0f;                           // sum of 4 uniforms: bell-shaped, in (-0.2, 0.2)
                for (int k = 0; k < 4; ++k) s += (float) (rng_next() >> 8) * (1.0f / 16777216.0f) - 0.5f;
                src[i] = s * 0.1f;
                imatrix[i] = 1.0f;
            }
            ggml_quantize_init_for(t->type);
            const int nblocks = N / t->block_elems;
            uint8_t * blocks = calloc((size_t) nblocks, bb);
            const size_t written = quantize(t->type, src, blocks, N, imatrix);
            if (written != (size_t) nblocks * bb) { fprintf(stderr, "harness: %s wrote %zu bytes, expected %zu\n", t->name, written, (size_t) nblocks * bb); return 2; }
            emit_case(t, "quantized", "llama.cpp quantization of 512 pseudo-random values in (-0.2, 0.2), imatrix of ones", blocks, nblocks);
            free(blocks);
        }

        // 4. MXFP4 only: the exponent byte at its limits.
        if (t->type == GGML_TYPE_MXFP4) {
            static const uint8_t exps[] = { 0, 1, 2, 126, 127, 128, 254, 255 };
            enum { NE = sizeof(exps) };
            uint8_t * blocks = malloc(NE * bb);
            for (int b = 0; b < NE; ++b) {
                blocks[b * bb] = exps[b];
                for (size_t i = 1; i < bb; ++i) blocks[b * bb + i] = (uint8_t) (rng_next() >> 24);
            }
            emit_case(t, "exponent_limits", "exponent byte 0, 1, 2, 126, 127, 128, 254, 255 with random values (255 overflows to infinity)", blocks, NE);
            free(blocks);
        }
        printf("\n    ]}");
    }
    printf("\n  ]\n}\n");
    return 0;
}
