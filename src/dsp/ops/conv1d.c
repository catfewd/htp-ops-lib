#include "dsp/hvx_internal.h"

int hvx_conv1d_f16(__fp16 *restrict dst, const __fp16 *restrict src,
                    const __fp16 *restrict weight, const __fp16 *restrict bias,
                    int T, int C_in, int C_out, int K, int stride, int pad) {
    if (!dst || !src || !weight || !bias || !T || !C_in || !C_out || !K || !stride) {
        return -1;
    }

    int T_out = (T + 2 * pad - K) / stride + 1;
    if (T_out <= 0) {
        return -1;
    }

    int n_ch_vecs = C_out / 64;
    int ch_rem    = C_out % 64;

    const HVX_Vector zero_v = Q6_V_vzero();

    // Process C_out in groups of 64 channels (one full HVX vector)
    for (int cg = 0; cg < n_ch_vecs; ++cg) {
        int c_off = cg * 64;

        for (int t = 0; t < T_out; ++t) {
            HVX_Vector acc_qf16 = Q6_Vqf16_vadd_VhfVhf(
                vmem((__fp16*)(bias + c_off)), zero_v);

            for (int k = 0; k < K; ++k) {
                int idx = t * stride + k - pad;
                if (idx < 0 || idx >= T) continue;

                const __fp16 *src_row  = src + idx * C_in;
                const __fp16 *w_base   = weight + k * C_in * C_out + c_off;

                for (int ci = 0; ci < C_in; ++ci) {
                    HVX_Vector vw = vmem((__fp16*)(w_base + ci * C_out));
                    __fp16 xs     = src_row[ci];
                    HVX_Vector vxs = Q6_Vh_vsplat_R(
                        (int32_t)fp16_to_bits(&xs));

                    acc_qf16 = Q6_Vqf16_vadd_Vqf16Vqf16(acc_qf16,
                        Q6_Vqf16_vmpy_VhfVhf(vw, vxs));
                }
            }

            vmem((__fp16*)(dst + t * C_out + c_off)) =
                Q6_Vhf_equals_Vqf16(acc_qf16);
        }
    }

    // Handle remainder channels (not a multiple of 64)
    if (ch_rem > 0) {
        int c_off = n_ch_vecs * 64;

        for (int t = 0; t < T_out; ++t) {
            for (int c = 0; c < ch_rem; ++c) {
                float sum = (float)bias[c_off + c];

                for (int k = 0; k < K; ++k) {
                    int idx = t * stride + k - pad;
                    if (idx < 0 || idx >= T) continue;

                    const __fp16 *src_row  = src + idx * C_in;
                    const __fp16 *w_plane  = weight + k * C_in * C_out + c_off + c;

                    for (int ci = 0; ci < C_in; ++ci) {
                        sum += (float)src_row[ci] * (float)w_plane[ci * C_out];
                    }
                }

                dst[t * C_out + c_off + c] = (__fp16)sum;
            }
        }
    }

    return 0;
}
