#include <math.h>

#include "dsp/hvx_internal.h"
#include "dsp/hvx_math.h"

#define VLEN_F32 32

static float hvx_hmax_f32(HVX_Vector v) {
    for (int s = VLEN_F32 / 2; s >= 1; s >>= 1) {
        HVX_Vector shifted = Q6_V_vlalign_VVR(v, Q6_V_vzero(), s * 4);
        v = Q6_Vsf_vmax_VsfVsf(v, shifted);
    }
    float tmp[VLEN_F32] __attribute__((aligned(VLEN)));
    vmem(tmp) = v;
    return tmp[0];
}

static float hvx_hsum_f32(HVX_Vector v) {
    HVX_Vector sum = Q6_Vqf32_vadd_VsfVsf(v, Q6_V_vzero());
    for (int s = VLEN_F32 / 2; s >= 1; s >>= 1) {
        sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_V_vlalign_VVR(sum, Q6_V_vzero(), s * 4));
    }
    float tmp[VLEN_F32] __attribute__((aligned(VLEN)));
    vmem(tmp) = Q6_Vsf_equals_Vqf32(sum);
    return tmp[0];
}

int hvx_soft_max_f32(float *restrict dst, const float *restrict src,
                      int ne00, int ne01, int ne02, int ne03) {
    if (!dst || !src || ne00 <= 0) return -1;

    int64_t n_rows = (int64_t)ne01 * ne02 * ne03;

    float log2e = 1.4426950408889634f;
    HVX_Vector log2e_v = Q6_V_vsplat_R(*(int32_t *)&log2e);
    HVX_Vector zero_v  = Q6_V_vzero();

    for (int64_t r = 0; r < n_rows; ++r) {
        float *restrict row_dst = dst + r * ne00;
        const float *restrict row_src = src + r * ne00;

        int nv = ne00 / VLEN_F32;
        int lv = ne00 % VLEN_F32;

        // find max
        float max_val = -INFINITY;
        if (nv > 0) {
            HVX_Vector vmax_sf = vmemu(row_src);
            for (int i = 1; i < nv; ++i) {
                HVX_Vector vs = vmemu(row_src + i * VLEN_F32);
                vmax_sf = Q6_Vsf_vmax_VsfVsf(vmax_sf, vs);
            }
            max_val = hvx_hmax_f32(vmax_sf);
        }
        for (int i = nv * VLEN_F32; i < ne00; ++i) {
            if (row_src[i] > max_val) max_val = row_src[i];
        }

        // compute exp(x - max) and sum
        HVX_Vector max_v = Q6_V_vsplat_R(*(int32_t *)&max_val);
        HVX_Vector sum_qf = zero_v;

        for (int i = 0; i < nv; ++i) {
            HVX_Vector xs = vmemu(row_src + i * VLEN_F32);

            // x - max (sf → qf32 → sf)
            HVX_Vector xmm_qf = Q6_Vqf32_vsub_VsfVsf(xs, max_v);
            HVX_Vector xmm_sf = Q6_Vsf_equals_Vqf32(xmm_qf);

            // exp(x-max) = exp2((x-max) * log2(e))
            HVX_Vector xl_qf = Q6_Vqf32_vmpy_VsfVsf(xmm_sf, log2e_v);
            HVX_Vector xl_sf = Q6_Vsf_equals_Vqf32(xl_qf);
            HVX_Vector ex_sf = hvx_my_exp2_vsf(xl_sf);

            vmemu(row_dst + i * VLEN_F32) = ex_sf;
            sum_qf = Q6_Vqf32_vadd_VsfVsf(sum_qf, ex_sf);
        }

        // leftover scalar
        float sum_val = 0.0f;
        for (int i = nv * VLEN_F32; i < ne00; ++i) {
            float xm = row_src[i] - max_val;
            float ex = expf(xm);
            row_dst[i] = ex;
            sum_val += ex;
        }

        // accumulate HVX sum lanes
        sum_val += hvx_hsum_f32(Q6_Vsf_equals_Vqf32(sum_qf));

        // divide each element by sum
        float inv_sum = 1.0f / sum_val;
        HVX_Vector inv_sum_v = Q6_V_vsplat_R(*(int32_t *)&inv_sum);

        for (int i = 0; i < nv; ++i) {
            HVX_Vector ev = vmemu(row_dst + i * VLEN_F32);
            vmemu(row_dst + i * VLEN_F32) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(ev, inv_sum_v));
        }
        for (int i = nv * VLEN_F32; i < ne00; ++i) {
            row_dst[i] *= inv_sum;
        }
    }

    return 0;
}
