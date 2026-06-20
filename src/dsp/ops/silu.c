#include <math.h>

#include "dsp/hvx_internal.h"
#include "dsp/hvx_math.h"

#define VLEN_F32 32

int hvx_silu_f32(float *restrict dst, const float *restrict src, int n) {
    if (!dst || !src || n <= 0) return -1;

    int n_vecs   = n / VLEN_F32;
    int leftover = n % VLEN_F32;

    float log2e        = 1.4426950408889634f;
    HVX_Vector log2e_v = Q6_V_vsplat_R(*(int32_t *)&log2e);
    HVX_Vector one_v   = Q6_V_vsplat_R(0x3F800000);
    HVX_Vector sign_v  = Q6_V_vsplat_R(0x80000000);

    for (int i = 0; i < n_vecs; ++i) {
        HVX_Vector x_sf = vmemu(src + i * VLEN_F32);

        // neg_x = -x (flip sign bit)
        HVX_Vector neg_x_sf = Q6_V_vxor_VV(x_sf, sign_v);

        // exp(-x) = exp2(-x * log2(e))
        HVX_Vector nxl_qf32  = Q6_Vqf32_vmpy_VsfVsf(neg_x_sf, log2e_v);
        HVX_Vector nxl_sf    = Q6_Vsf_equals_Vqf32(nxl_qf32);
        HVX_Vector exp_nx_sf = hvx_my_exp2_vsf(nxl_sf);

        // 1 + exp(-x)
        HVX_Vector denom_qf32 = Q6_Vqf32_vadd_VsfVsf(one_v, exp_nx_sf);
        HVX_Vector denom_sf   = Q6_Vsf_equals_Vqf32(denom_qf32);

        // sigmoid = 1/(1+exp(-x))
        HVX_Vector sig_qf32 = hvx_my_inv_vqf32_vsf(denom_sf);
        HVX_Vector sig_sf   = Q6_Vsf_equals_Vqf32(sig_qf32);

        // x * sigmoid(x)
        HVX_Vector out_qf32 = Q6_Vqf32_vmpy_VsfVsf(x_sf, sig_sf);
        vmemu(dst + i * VLEN_F32) = Q6_Vsf_equals_Vqf32(out_qf32);
    }

    for (int i = 0; i < leftover; ++i) {
        int idx  = n_vecs * VLEN_F32 + i;
        float x  = src[idx];
        float sg = 1.0f / (1.0f + expf(-x));
        dst[idx] = x * sg;
    }

    return 0;
}
