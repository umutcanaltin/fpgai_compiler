#include "external_wrapper.h"
#include <cmath>
#include <cstdio>

int main() {
    float input[4];
    float output[4] = {0};
    for (int i = 0; i < 4; ++i) input[i] = (float)(i - 2);
    deeplearn(input, output);
    int failures = 0;
    for (int i = 0; i < 4; ++i) {
        float expected = input[i] * (float)2.0f + (float)1.0f;
        if (std::fabs((double)(output[i] - expected)) > 1.0e-6) ++failures;
    }
    std::printf("[FPGAI-EXTERNAL-HLS] failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}
