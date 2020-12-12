#include "mul256.cu"

// Computes Z = X*X mod P times count
__device__ __forceinline__ void sq_times(uint32* Y, const uint32* X, int count)
{
    uint32 T[8];
    copy(T, X);

    do {
        mul_reduce_2(Y, T, T);
        copy(T, Y);
    } while (--count);
}

__device__ __forceinline__ void invert(uint32* Z){
    t256 a, b, c, t;

    copy(c.u32, Z);

    sq_times(a.u32, c.u32, 1);
    sq_times(t.u32, a.u32, 2);
    mul_reduce(b.u32, c.u32, t.u32);
    mul_reduce(a.u32, b.u32, a.u32);
    sq_times(t.u32, a.u32, 1);
    mul_reduce(b.u32, t.u32, b.u32);
    sq_times(t.u32, b.u32, 5);
    mul_reduce(b.u32, t.u32, b.u32);
    sq_times(t.u32, b.u32, 10);
    mul_reduce(c.u32, b.u32, t.u32);
    sq_times(t.u32, c.u32, 20);
    mul_reduce(t.u32, c.u32, t.u32);
    sq_times(t.u32, t.u32, 10);
    mul_reduce(b.u32, t.u32, b.u32);
    sq_times(t.u32, b.u32, 50);
    mul_reduce(c.u32, b.u32, t.u32);
    sq_times(t.u32, c.u32, 100);
    mul_reduce(t.u32, c.u32, t.u32);
    sq_times(t.u32, t.u32, 50);
    mul_reduce(b.u32, t.u32, b.u32);
    sq_times(b.u32, b.u32, 5);
    mul_reduce(b.u32, a.u32, b.u32);

    copy(Z, b.u32);
}