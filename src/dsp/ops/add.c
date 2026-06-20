#include "dsp/hvx_internal.h"

#define VLEN_F32 32

int hvx_add_f32(float *restrict dst, const float *restrict a, const float *restrict b, int n) {
    if (!dst || !a || !b || n <= 0) return -1;

    int n_vecs   = n / VLEN_F32;
    int leftover = n % VLEN_F32;

    for (int i = 0; i < n_vecs; ++i) {
        HVX_Vector va_sf = vmemu(a + i * VLEN_F32);
        HVX_Vector vb_sf = vmemu(b + i * VLEN_F32);
        HVX_Vector sum   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(va_sf, vb_sf));
        vmemu(dst + i * VLEN_F32) = sum;
    }

    for (int i = 0; i < leftover; ++i) {
        int idx = n_vecs * VLEN_F32 + i;
        dst[idx] = a[idx] + b[idx];
    }

    return 0;
}
