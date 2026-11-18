/* The C compiler checks all SDK signatures against the pinned upstream headers. */
#include <float.h>
#include <stddef.h>
#include <stdint.h>
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod.h"

size_t quantinfer_kai_lhs_size(size_t m,size_t k) { return kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(m,k,32,4,8,2); }
size_t quantinfer_kai_rhs_size(size_t n,size_t k) { return kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(n,k,4,8,32); }
void quantinfer_kai_lhs_pack(size_t m,size_t k,const float *lhs,void *packed) {
    kai_run_lhs_quant_pack_qsi8d32p_f32(m,k,32,4,8,2,0,lhs,k*sizeof(float),packed);
}
void quantinfer_kai_rhs_pack(size_t n,size_t k,const uint8_t *rhs,void *packed) {
    const struct kai_rhs_pack_qs4cxs1s0_param params={1,8};
    kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(1,n,k,4,8,2,32,rhs,NULL,packed,0,&params);
}
void quantinfer_kai_run(size_t m,size_t n,size_t k,const void *lhs,const void *rhs,float *output) {
    kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod(m,n,k,32,lhs,rhs,output,n*sizeof(float),sizeof(float),-FLT_MAX,FLT_MAX);
}
