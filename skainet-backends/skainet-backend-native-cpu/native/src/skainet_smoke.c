#include "skainet_kernels.h"

void skainet_smoke_double(const float* input, float* output, int32_t length) {
    for (int32_t i = 0; i < length; ++i) {
        output[i] = 2.0f * input[i];
    }
}
