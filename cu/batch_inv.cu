#include "sq256.cu"
 
#ifndef BLOCKS
#define BLOCKS 3584*16
#endif
  
#ifndef N_BATCH
#define N_BATCH 256
#endif

typedef t256 batch_z[BLOCKS];
typedef t256 t512_batch[BLOCKS*N_BATCH*2]; 

__global__ void batch_inv_step0(batch_z *a, batch_z *w)
{ 
    t256 c0, d0;  

    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    
    copy(c0.u32, a[0][gid].u32);
    copy(w[0][gid].u32, c0.u32);

    for(int i = 1; i < N_BATCH; i++){
        copy(d0.u32, a[i][gid].u32);

        mul_reduce(c0.u32, d0.u32, c0.u32);

        copy(w[i][gid].u32, c0.u32);
    }
}               
                                                                  
__global__ void batch_inv_step1(batch_z *w)
{
    t256 in;

    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
 
    copy(in.u32, w[N_BATCH-1][gid].u32); 
    
    invert(in.u32);
       
    copy(w[N_BATCH-1][gid].u32, in.u32);
}

__global__ void batch_inv_step2(batch_z *z, batch_z *inv_z, batch_z *w)
{ 
    t256 c, t, v;

    const int gid = blockDim.x * blockIdx.x + threadIdx.x;

    copy(c.u32, w[N_BATCH-1][gid].u32);

    for(int i = N_BATCH-1;i; i--){
        copy(t.u32, w[i-1][gid].u32);
        mul_reduce(t.u32, c.u32, t.u32);

        copy(v.u32, z[i][gid].u32);
        copy(inv_z[i][gid].u32, t.u32);

        mul_reduce(c.u32, v.u32, c.u32);
    }

    copy(inv_z[0][gid].u32, c.u32);
} 


__global__ void calc_next_optimized_point(t256 *xmy, t256 *xpy, t256 *t, t512_batch xy, t256 *inv_z)
{
    t256 e, f, out_xmy, out_xpy, z_in;

    const uint32 inv_I[8] = {0xb5f15f3d, 0x3b11e4d8, 0x52d01b87, 0xd0bce7f9, 0xc2042858, 0xd4b2ff66, 0xb03e20f4, 0x547cdb7f};

    const int globalid = (blockDim.x * blockIdx.x + threadIdx.x);

    copy(z_in.u32, inv_z[((N_BATCH*globalid)+(N_BATCH-1))].u32);
    copy(e.u32, xy[((N_BATCH*globalid)+(N_BATCH-1))].u32); // x * I

    mul_mod_2(e.u32, z_in.u32, e.u32);
    mul_mod_2(e.u32, inv_I, e.u32); //  x * I * inv(I)

    copy(f.u32, xy[((BLOCKS*N_BATCH)+(N_BATCH*globalid)+(N_BATCH-1))].u32); // y

    mul_mod_2(f.u32, z_in.u32, f.u32);

    sub_reduce(out_xmy.u32, f.u32, e.u32); // y - x
    copy(xmy[globalid].u32, out_xmy.u32);

    add_reduce(out_xpy.u32, f.u32, e.u32); // x + y
    copy(xpy[globalid].u32, out_xpy.u32);

    mul_mod_2(f.u32, e.u32, f.u32);  // t = x * y
    copy(t[globalid].u32, f.u32);
}  

__global__ void test_batch_inv(t256 *w, t256 *out_test)
{
    t256 in, out;

    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
 
    copy(in.u32, w[gid].u32); 
      
    bn_mod_inverse(out.u32, in.u32);
       
    copy(w[gid].u32, out.u32);
}
