#include "sq256.cu"

#ifndef N_BATCH
#define N_BATCH 256
#endif  

__device__ __forceinline__ void Folds8(uint8 *Y, const uint32 *X)
{
    int i, j;
    uint8 a = 0;

    for (i = 32; i-- > 0;)
    {
        for (j = 8; j-- > 0;) a = (a << 1) + ((X[j] >> i) & 1);
        *Y++ = a;
    }
}

__device__ __forceinline__ void SetValue(uint32* X, uint32 value)
{
    X[0] = value;
    X[1] = X[2] = X[3] = X[4] = X[5] = X[6] = X[7] = 0;
}

// From https://github.com/msotoodeh/curve25519
__global__ void scalarmult_b(const t256 *const_fold8_ypx, const t256 *const_fold8_ymx, const t256 *const_fold8_t, t256 *out_x, t256 *out_y, t256 *out_t, const t256 *secret_key)
{
	__shared__ t256 s_fold8_ypx[N_BATCH], s_fold8_ymx[N_BATCH], s_fold8_t[N_BATCH];
	__shared__ uint8 cut[N_BATCH][32];

	t256 sk, a, b, c, d, e, f, g, h, t, v;
	t256 Sx, Sy, Sz, St;
	t256 p0, p1;

    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const int lid = threadIdx.x;

 	copy(s_fold8_ypx[lid].u32, const_fold8_ypx[lid].u32);
 	copy(s_fold8_ymx[lid].u32, const_fold8_ymx[lid].u32);
 	copy(s_fold8_t[lid].u32, const_fold8_t[lid].u32);
    __syncthreads();

    copy(sk.u32, secret_key[gid].u32);
    Folds8(cut[lid], sk.u32);

    copy(p0.u32, s_fold8_ypx[cut[lid][0]].u32);
    copy(p1.u32, s_fold8_ymx[cut[lid][0]].u32);

    sub_mod(Sx.u32, p0.u32, p1.u32);
    add_mod(Sy.u32, p0.u32, p1.u32);

    SetValue(Sz.u32, 2);
 
    for(int i = 1; i < 32; i++){
        // Double 
        mul_mod(a.u32, Sx.u32, Sx.u32);      //  a = x1*x1 % q
        mul_mod(b.u32, Sy.u32, Sy.u32);      //  b = y1*y1 % q
  
        mul_mod(c.u32, Sz.u32, Sz.u32);      // c = z1*z1 % q
        add_mod(c.u32, c.u32, c.u32);        // c = 2*c % q
        neg256_mod(d.u32, a.u32);            // d = -a % q

        sub_mod(h.u32, d.u32, b.u32);       // h = d - b % q
        sub_mod(g.u32, b.u32, a.u32);       // g = b - a % q
        sub_mod(f.u32, g.u32, c.u32);       // f = g - c % q
        add_mod(t.u32, Sx.u32, Sy.u32);     // t = x1+y1 % q
        mul_mod(v.u32, t.u32, t.u32);     // v = t*t % q
        add_mod(e.u32, v.u32, h.u32);       // e = (v + h) % q

        mul_mod(Sx.u32, e.u32, f.u32);       // x3 = e*f % q
        mul_mod(Sy.u32, h.u32, g.u32);       // y3 = g*h % q
        mul_mod(Sz.u32, g.u32, f.u32);       // z3 = f*g % q
        mul_mod(St.u32, e.u32, h.u32);       // t3 = e*h % q

        // Add
        sub_mod(a.u32, Sy.u32, Sx.u32);                           // a = (y1-x1) % q
        mul_mod(a.u32, a.u32, s_fold8_ymx[cut[lid][i]].u32);      // a = a*y2 % q
        add_mod(b.u32, Sy.u32, Sx.u32);                           // b = (y1+x1) % q
        mul_mod(b.u32, b.u32, s_fold8_ypx[cut[lid][i]].u32);      // b = b*x2 % q
        mul_mod(c.u32, St.u32, s_fold8_t[cut[lid][i]].u32);       // c = t1*t2 % q
        add_mod(d.u32, Sz.u32, Sz.u32);                           // d = (z1+z1) % q

        sub_mod(e.u32, b.u32, a.u32);      // e = b - a % q
        sub_mod(f.u32, d.u32, c.u32);      // f = d - c % q
        add_mod(g.u32, d.u32, c.u32);      // g = d + c % q
        add_mod(h.u32, b.u32, a.u32);      // h = b + a % q

        mul_mod(Sx.u32, e.u32, f.u32);      // x3 = e*f % q
        mul_mod(Sy.u32, h.u32, g.u32);      // y3 = h*g % q
        mul_mod(Sz.u32, g.u32, f.u32);      // z3 = g*f % q
	}

    invert(Sz.u32);

    mul_mod(Sx.u32, Sx.u32, Sz.u32);
    mul_mod(Sy.u32, Sy.u32, Sz.u32);

    mul_mod(St.u32, Sx.u32, Sy.u32);

    copy(out_t[gid].u32, St.u32);

    sub_mod(a.u32, Sy.u32, Sx.u32);
    copy(out_x[gid].u32, a.u32);

    add_mod(b.u32, Sy.u32, Sx.u32);
	copy(out_y[gid].u32, b.u32);
	
}