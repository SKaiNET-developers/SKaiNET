/*
 * Row-range threading shared by the packed matmul kernels (#1195).
 *
 * The packed kernels parallelize over OUTPUT ROWS: participants pull disjoint
 * `[o_start, o_end)` grains of the output vector, so there are no
 * accumulation races and no synchronization beyond the completion barrier.
 * Per output row the accumulation order over input blocks is unchanged, and
 * a row's result does not depend on which thread computes it — threaded
 * results are bit-identical to single-threaded ones (the parity suites are
 * the oracle for that claim).
 *
 * Execution model, shaped by a series of Pixel 8a measurements (#1195, all
 * Qwen2.5-1.5B Q4_K_M decode steps; single-threaded baseline 153 ms):
 *
 * 1. POOL, NOT CREATE/JOIN. The obvious create/join-per-call variant ran
 *    994 ms/step: ~600 pthread_create per step against cores in deep cpuidle
 *    pays milliseconds of wakeup latency each. Workers are created once,
 *    lazily (pthread_once), and live for the process.
 *
 * 2. SPIN BRIEFLY, THEN PARK. A pool whose workers sleep between jobs still
 *    measured only 115–306 ms/step across chunking variants — sub-millisecond
 *    parallel bursts separated by sleeps never accumulate per-thread
 *    utilization, so EAS/schedutil keeps the workers on little cores at low
 *    clocks while the single-threaded caller would have pegged one big core
 *    at max clock. Workers therefore spin (`yield`) on the job epoch for
 *    ~SKAINET_SPIN_ITERS before parking on the condvar: during a decode the
 *    gaps between matmuls are far shorter than the spin window, utilization
 *    stays pegged, and the scheduler answers with big cores and full clocks
 *    (the same reason llama.cpp's thread pool spins). Once work stops
 *    arriving, everyone parks — no battery burn at idle.
 *
 * 3. GUIDED grains — remaining/(2·parts), floored at SKAINET_MATMUL_GRAIN —
 *    off an atomic cursor: long contiguous streams first (prefetch-friendly),
 *    shrinking toward the tail so a straggler holds a small tail rather than
 *    a quarter of the matrix. On symmetric cores this degrades to an even
 *    split; nothing here is tuned to one SoC's topology.
 *
 * Threading engages only when `n` reaches the threshold — below it fixed
 * costs dominate any win, and tiny projections (e.g. GQA k/v with output_dim
 * 256) stay single-threaded on purpose. A concurrent second caller while the
 * pool is busy simply runs its rows on its own thread (correct, unshared);
 * so does everything if worker creation ever failed.
 *
 * MSVC has no <pthread.h>; there the runner degrades to a plain call on the
 * caller's thread.
 */
#include "skainet_row_threads.h"

#if !defined(_MSC_VER)

#include <pthread.h>
#include <stdatomic.h>

#define SKAINET_POOL_WORKERS (SKAINET_MATMUL_THREADS - 1)
#define SKAINET_MATMUL_GRAIN 64
/* ~a millisecond of `yield`s — longer than the gaps between a decode step's
 * matmul calls, far shorter than "the model stopped decoding". */
#define SKAINET_SPIN_ITERS (1u << 20)

#if defined(__aarch64__) || defined(__arm__)
#define SKAINET_CPU_RELAX() __asm__ __volatile__("yield" ::: "memory")
#elif defined(__x86_64__) || defined(__i386__)
#define SKAINET_CPU_RELAX() __asm__ __volatile__("pause" ::: "memory")
#else
#define SKAINET_CPU_RELAX() ((void) 0)
#endif

typedef struct {
    pthread_mutex_t m;          /* serializes callers; guards the park/wake handoffs */
    pthread_cond_t cv_work;
    pthread_cond_t cv_done;
    skainet_row_range_fn fn;    /* job fields: written before the epoch release-store */
    void* ctx;
    int32_t n;
    int parts;
    atomic_int_fast32_t cursor; /* next unclaimed row of the current job */
    atomic_ulong epoch;         /* release-published per job; the workers' work signal */
    atomic_int remaining;       /* workers not yet finished with the current job */
    atomic_int work_waiters;    /* workers parked on cv_work (broadcast only then) */
    atomic_int done_waiter;     /* caller parked on cv_done (signal only then) */
    int busy;                   /* a job is in flight (second callers go solo) */
    int workers_alive;
} skainet_row_pool;

static skainet_row_pool skainet_g_row_pool = {
    PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER,
    NULL, NULL, 0, 0, 0, 0UL, 0, 0, 0, 0, 0,
};
static pthread_once_t skainet_g_row_pool_once = PTHREAD_ONCE_INIT;

/* Guided sizing (OpenMP `schedule(guided)` shape): see file header, point 3. */
static void skainet_row_pool_drain(skainet_row_range_fn fn, void* ctx, int32_t n, int parts) {
    int_fast32_t cur = atomic_load_explicit(&skainet_g_row_pool.cursor, memory_order_relaxed);
    for (;;) {
        if ((int32_t) cur >= n) return;
        int32_t want = (n - (int32_t) cur) / (2 * parts);
        if (want < SKAINET_MATMUL_GRAIN) want = SKAINET_MATMUL_GRAIN;
        if (atomic_compare_exchange_weak_explicit(
                &skainet_g_row_pool.cursor, &cur, cur + want,
                memory_order_relaxed, memory_order_relaxed)) {
            const int32_t s = (int32_t) cur;
            int32_t e = s + want;
            if (e > n) e = n;
            fn(ctx, s, e);
            cur = atomic_load_explicit(&skainet_g_row_pool.cursor, memory_order_relaxed);
        }
        /* CAS failure reloaded `cur`; loop retries with the fresh value. */
    }
}

static void* skainet_row_pool_worker(void* arg) {
    (void) arg;
    unsigned long seen = 0UL;
    for (;;) {
        /* Spin for the next job; park only when none arrives in the window. */
        unsigned spins = 0;
        while (atomic_load_explicit(&skainet_g_row_pool.epoch, memory_order_acquire) == seen) {
            if (++spins >= SKAINET_SPIN_ITERS) {
                pthread_mutex_lock(&skainet_g_row_pool.m);
                atomic_fetch_add_explicit(&skainet_g_row_pool.work_waiters, 1, memory_order_relaxed);
                while (atomic_load_explicit(&skainet_g_row_pool.epoch, memory_order_acquire) == seen) {
                    pthread_cond_wait(&skainet_g_row_pool.cv_work, &skainet_g_row_pool.m);
                }
                atomic_fetch_sub_explicit(&skainet_g_row_pool.work_waiters, 1, memory_order_relaxed);
                pthread_mutex_unlock(&skainet_g_row_pool.m);
                break;
            }
            SKAINET_CPU_RELAX();
        }
        seen = atomic_load_explicit(&skainet_g_row_pool.epoch, memory_order_acquire);

        /* Job fields were written before the epoch release-store — the acquire
         * above orders these plain reads, and they are stable while busy. */
        skainet_row_pool_drain(skainet_g_row_pool.fn, skainet_g_row_pool.ctx,
                               skainet_g_row_pool.n, skainet_g_row_pool.parts);

        if (atomic_fetch_sub_explicit(&skainet_g_row_pool.remaining, 1, memory_order_acq_rel) == 1) {
            /* Last one out: the park/wake handoff must be decided under the
             * mutex — a bare done_waiter load could miss a caller that is
             * between setting the flag and blocking, and sleep it forever.
             * Cost: one lock/unlock per job, by one thread. */
            pthread_mutex_lock(&skainet_g_row_pool.m);
            if (atomic_load_explicit(&skainet_g_row_pool.done_waiter, memory_order_relaxed) != 0) {
                pthread_cond_signal(&skainet_g_row_pool.cv_done);
            }
            pthread_mutex_unlock(&skainet_g_row_pool.m);
        }
    }
    /* unreachable */
}

static void skainet_row_pool_init(void) {
    for (int i = 0; i < SKAINET_POOL_WORKERS; ++i) {
        pthread_t t;
        if (pthread_create(&t, NULL, skainet_row_pool_worker, NULL) != 0) {
            break; /* fewer participants; the cursor still covers every row */
        }
        pthread_detach(t);
        ++skainet_g_row_pool.workers_alive;
    }
}

/*
 * Run `fn` over rows [0, n): every participant (workers + the calling
 * thread) pulls guided grains from the cursor; the caller then spins briefly
 * on the completion count before parking. Single-threaded when n is under
 * the threshold, when the pool is busy with another caller's job, or when no
 * workers exist.
 */
void skainet_run_rows(skainet_row_range_fn fn, void* ctx, int32_t n) {
    if (n <= 0) return;
    if (n < SKAINET_MATMUL_THREAD_THRESHOLD) {
        fn(ctx, 0, n);
        return;
    }
    pthread_once(&skainet_g_row_pool_once, skainet_row_pool_init);

    pthread_mutex_lock(&skainet_g_row_pool.m);
    if (skainet_g_row_pool.workers_alive == 0 || skainet_g_row_pool.busy) {
        pthread_mutex_unlock(&skainet_g_row_pool.m);
        fn(ctx, 0, n);
        return;
    }
    skainet_g_row_pool.busy = 1;
    skainet_g_row_pool.fn = fn;
    skainet_g_row_pool.ctx = ctx;
    skainet_g_row_pool.n = n;
    skainet_g_row_pool.parts = skainet_g_row_pool.workers_alive + 1;
    atomic_store_explicit(&skainet_g_row_pool.cursor, 0, memory_order_relaxed);
    atomic_store_explicit(&skainet_g_row_pool.remaining, skainet_g_row_pool.workers_alive,
                          memory_order_relaxed);
    /* Publish: job fields above happen-before this release-store. */
    atomic_store_explicit(&skainet_g_row_pool.epoch,
                          atomic_load_explicit(&skainet_g_row_pool.epoch, memory_order_relaxed) + 1UL,
                          memory_order_release);
    if (atomic_load_explicit(&skainet_g_row_pool.work_waiters, memory_order_relaxed) != 0) {
        pthread_cond_broadcast(&skainet_g_row_pool.cv_work);
    }
    const int parts = skainet_g_row_pool.parts;
    pthread_mutex_unlock(&skainet_g_row_pool.m);

    skainet_row_pool_drain(fn, ctx, n, parts);

    /* Spin briefly for the stragglers' tail, then park. */
    unsigned spins = 0;
    while (atomic_load_explicit(&skainet_g_row_pool.remaining, memory_order_acquire) > 0) {
        if (++spins >= SKAINET_SPIN_ITERS) {
            pthread_mutex_lock(&skainet_g_row_pool.m);
            atomic_store_explicit(&skainet_g_row_pool.done_waiter, 1, memory_order_release);
            while (atomic_load_explicit(&skainet_g_row_pool.remaining, memory_order_acquire) > 0) {
                pthread_cond_wait(&skainet_g_row_pool.cv_done, &skainet_g_row_pool.m);
            }
            atomic_store_explicit(&skainet_g_row_pool.done_waiter, 0, memory_order_relaxed);
            pthread_mutex_unlock(&skainet_g_row_pool.m);
            break;
        }
        SKAINET_CPU_RELAX();
    }

    pthread_mutex_lock(&skainet_g_row_pool.m);
    skainet_g_row_pool.busy = 0;
    pthread_mutex_unlock(&skainet_g_row_pool.m);
}

#else /* MSVC: no pthreads — single-threaded, same numerics */

void skainet_run_rows(skainet_row_range_fn fn, void* ctx, int32_t n) {
    if (n > 0) fn(ctx, 0, n);
}

#endif
