#include "skainet_cpu_features.h"

#if defined(__APPLE__) && defined(__aarch64__)

#include <sys/sysctl.h>

int skainet_cpu_has_dotprod(void) {
    /* Benign-race cache: concurrent first calls compute and write the same
     * value, so no synchronization is needed. */
    static int cached = -1;
    if (cached < 0) {
        int v = 0;
        size_t sz = sizeof v;
        /* The key exists since iOS 15 / macOS 12; on older OS versions
         * sysctlbyname fails => 0 => scalar fallback (always safe). A12
         * (iPhone XS/XR) reports 0; A13+ and every Apple Silicon Mac
         * report 1. */
        if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &v, &sz, NULL, 0) != 0) {
            v = 0;
        }
        cached = (v != 0);
    }
    return cached;
}

#else /* non-Apple: the compile-time -march decides, no runtime probe. */

int skainet_cpu_has_dotprod(void) {
#if defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}

#endif
