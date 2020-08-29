#include "mul256.cu"

#define GID (blockDim.x * blockIdx.x + threadIdx.x)

#ifndef BLOCKS
#define BLOCKS 3584*16
#endif

#ifndef N_BATCH
#define N_BATCH 256
#endif  

typedef t256 t1024_batch[BLOCKS*N_BATCH*4];

__device__ __constant__ uint32 I[] =
{
    0x4a0ea0b0, 0xc4ee1b27, 0xad2fe478, 0x2f431806,
    0x3dfbd7a7, 0x2b4d0099, 0x4fc1df0b, 0x2b832480,
};

__global__ void batch_ge_add_step0(const t256 *const_b_table_xpy, const t256 *const_b_table_ymx, const t256 *const_b_table_t, const t256 *ymx, const t256 *xpy , const t256 *t, t1024_batch out_xy, t256 *temp_c)
{
	__shared__ t256 s_b_table[N_BATCH*2], ymx_l, xpy_l;
    t256 A, B, out, c1, c2;

    const int globalid1 = (GID < (BLOCKS*N_BATCH)) ? GID : ((BLOCKS * N_BATCH)*2 + GID);
    const int globalid2 = (GID < (BLOCKS*N_BATCH)) ? GID : (GID - (BLOCKS * N_BATCH));

    const int c_id1 = (BLOCKS * N_BATCH) + GID;

    const int lid = threadIdx.x;
 	
    const int lid1 = (GID < (BLOCKS*N_BATCH)) ? (lid+N_BATCH) : lid;
    const int lid2 = (GID < (BLOCKS*N_BATCH)) ? lid : (lid+N_BATCH);

    const int gid2 = (GID < (BLOCKS*N_BATCH)) ? blockIdx.x : (blockIdx.x-BLOCKS);

    copy(s_b_table[lid].u32, const_b_table_xpy[lid].u32);
    copy(s_b_table[N_BATCH+lid].u32, const_b_table_ymx[lid].u32);
    copy(ymx_l.u32, ymx[gid2].u32);
    copy(xpy_l.u32, xpy[gid2].u32);
    copy(c1.u32, temp_c[globalid2].u32); // C = 2 + C
    __syncthreads();

 	mul_reduce(A.u32, s_b_table[lid1].u32, ymx_l.u32); // A1 = (Y1-X1)*(Y2-X2) / A2 = (Y1+X1)*(Y2-X2)
    mul_reduce(B.u32, s_b_table[lid2].u32, xpy_l.u32); // B1 = (Y1+X1)*(Y2+X2) / B2 = (Y1-X1)*(Y2+X2) 
    sub_4(c2.u32, c1.u32); // F = 4 - C

    s_b_table[lid].u256 = (GID < (BLOCKS*N_BATCH)) ? c1.u256 : c2.u256;
    s_b_table[N_BATCH+lid].u256 = (GID < (BLOCKS*N_BATCH)) ? c2.u256 : c1.u256;
    __syncthreads();

    add_reduce(out.u32, B.u32, A.u32); // H = B + A
    mul_reduce(out.u32, s_b_table[lid].u32, out.u32); // Y3 = H * G / Y4 = H * F
    copy(out_xy[c_id1].u32, out.u32);

    sub_reduce(out.u32, B.u32, A.u32); // E = B - A
    mul_reduce(out.u32, s_b_table[N_BATCH+lid].u32, out.u32); // X3 = E * F * I / X4 = E * G * I / I = SQRT(-1)
    mul_reduce(out.u32, I, out.u32);
    copy(out_xy[globalid1].u32, out.u32);
}  

__global__ void batch_ge_add_step1(const t256 *const_b_table_t, const t256 *t, t1024_batch out_xy, t256 *in_z, t256 *temp_c)
{
	__shared__ t256 s_b_table_t[N_BATCH], s_t;
	t256 c, z_out;

 	copy(s_b_table_t[threadIdx.x].u32, const_b_table_t[threadIdx.x].u32);
    copy(s_t.u32, t[blockIdx.x].u32);
    __syncthreads();

    mul_reduce(c.u32, s_b_table_t[threadIdx.x].u32, s_t.u32); // C = T1*k*T2
    mul_reduce(z_out.u32, c.u32, c.u32); // z_out = C ^ 2
    sub_4(z_out.u32, z_out.u32);         // z_out = 4 - C ^ 2
    copy(in_z[GID].u32, z_out.u32);

    add_2(c.u32, c.u32); // C = 2 + C
    copy(temp_c[GID].u32, c.u32);
}