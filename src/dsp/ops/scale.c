#include "dsp/hvx_internal.h"
#include "dsp/hvx_math.h"

#define VLEN_F32 32

int hvx_scale_f32(float *restrict dst, const float *restrict src, float scale, int n) {
    if (!dst || !src || n <= 0) return -1;

    int n_vecs   = n / VLEN_F32;
    int leftover = n % VLEN_F32;

    HVX_Vector scale_v = Q6_V_vsplat_R(*(int32_t *)&scale);

    for (int i = 0; i < n_vecs; ++i) {
        HVX_Vector vs_sf  = vmemu(src + i * VLEN_F32);
        HVX_Vector prod   = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(vs_sf, scale_v));
        vmemu(dst + i * VLEN_F32) = prod;
    }

    for (int i = 0; i < leftover; ++i) {
        int idx = n_vecs * VLEN_F32 + i;
        dst[idx] = src[idx] * scale;
    }

    return 0;
}
