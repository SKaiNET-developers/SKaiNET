/*
 * Row-range threading shared by the packed matmul kernels (#1195) — see
 * skainet_row_threads.c for the pool and the measured rationale.
 */
#ifndef SKAINET_ROW_THREADS_H
#define SKAINET_ROW_THREADS_H

#include <stdint.h>

#define SKAINET_MATMUL_THREADS          4
#define SKAINET_MATMUL_THREAD_THRESHOLD 512

typedef void (*skainet_row_range_fn)(void* ctx, int32_t o_start, int32_t o_end);

/*
 * Run `fn` over rows [0, n): threaded over the shared worker pool when
 * n >= SKAINET_MATMUL_THREAD_THRESHOLD, on the calling thread otherwise
 * (also on MSVC, when the pool is busy with another caller, or when worker
 * creation failed). Bit-identical to the single-threaded result: workers own
 * disjoint [o_start, o_end) row ranges and per-row accumulation order never
 * changes.
 */
void skainet_run_rows(skainet_row_range_fn fn, void* ctx, int32_t n);

#endif /* SKAINET_ROW_THREADS_H */
