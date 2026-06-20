#pragma once

#include <stdint.h>

enum HtpOpsIndex {
  HTP_OPS_RMS_NORM_F32,
  HTP_OPS_MAT_MUL_PERMUTED_W16A32,
  HTP_OPS_MAT_MUL_PERMUTED_W4D16A32,
  HTP_OPS_MAT_MUL_PERMUTED_W8D16A32,
  HTP_OPS_MAT_MUL_PERMUTED_W4D16A32_IQ4_NL,
  HTP_OPS_FLASH_ATTN_QO_F32_KV_F16,
  HTP_OPS_CONV1D_F32,
  HTP_OPS_CONV1D_F16,
  HTP_OPS_SILU_F32,
  HTP_OPS_ADD_F32,
  HTP_OPS_MUL_F32,
  HTP_OPS_SOFT_MAX_F32,
  HTP_OPS_SCALE_F32,
  HTP_OPS_COPY_F32,
  HTP_OPS_COUNT,
};

struct RpcmemBufAddr {
  int32_t fd;
  int32_t offset;
} __attribute__((packed));

struct RmsNormF32Params {
  struct RpcmemBufAddr dst;
  struct RpcmemBufAddr src;
  int32_t       ne0;
  int32_t       ne1;
} __attribute__((packed));

struct MatMulParams {
  struct RpcmemBufAddr output;
  struct RpcmemBufAddr activation; // m * k
  struct RpcmemBufAddr weight; // k * n
  int32_t m;
  int32_t k;
  int32_t n;
} __attribute__((packed));

struct FlashAttnParams {
  struct RpcmemBufAddr o;
  struct RpcmemBufAddr q;
  struct RpcmemBufAddr k;
  struct RpcmemBufAddr v;
  struct RpcmemBufAddr mask;
  int32_t qo_len;
  int32_t kv_len;
  int32_t n_heads;
  int32_t n_kv_heads;
  int32_t head_dim;
} __attribute__((packed));

struct Conv1dF16Params {
  struct RpcmemBufAddr dst;
  struct RpcmemBufAddr src;
  struct RpcmemBufAddr weight;
  struct RpcmemBufAddr bias;
  int32_t T;
  int32_t C_in;
  int32_t C_out;
  int32_t K;
  int32_t stride;
  int32_t pad;
} __attribute__((packed));

struct SiluF32Params {
  struct RpcmemBufAddr dst;
  struct RpcmemBufAddr src;
  int32_t       ne;
} __attribute__((packed));

struct AddF32Params {
  struct RpcmemBufAddr dst;
  struct RpcmemBufAddr a;
  struct RpcmemBufAddr b;
  int32_t       ne;
} __attribute__((packed));

struct MulF32Params {
  struct RpcmemBufAddr dst;
  struct RpcmemBufAddr a;
  struct RpcmemBufAddr b;
  int32_t       ne;
} __attribute__((packed));

struct SoftMaxF32Params {
  struct RpcmemBufAddr dst;
  struct RpcmemBufAddr src;
  int32_t       ne00;
  int32_t       ne01;
  int32_t       ne02;
  int32_t       ne03;
} __attribute__((packed));

struct ScaleF32Params {
  struct RpcmemBufAddr dst;
  struct RpcmemBufAddr src;
  float         scale;
  int32_t       ne;
} __attribute__((packed));
