#include "mul256.cu"

/*
 * Bitwise shift
 */
__device__ __forceinline__ void bn_rshift1_2(uint32 *bna, uint32 *bnb)
{
    bna[0] = (bna[1] << 31) | (bna[0] >> 1);
    bna[1] = (bna[2] << 31) | (bna[1] >> 1);
    bna[2] = (bna[3] << 31) | (bna[2] >> 1);
    bna[3] = (bna[4] << 31) | (bna[3] >> 1);
    bna[4] = (bna[5] << 31) | (bna[4] >> 1);
    bna[5] = (bna[6] << 31) | (bna[5] >> 1);
    bna[6] = (bna[7] << 31) | (bna[6] >> 1);
    bna[7] >>= 1;

    bnb[0] = (bnb[1] << 31) | (bnb[0] >> 1);
    bnb[1] = (bnb[2] << 31) | (bnb[1] >> 1);
    bnb[2] = (bnb[3] << 31) | (bnb[2] >> 1);
    bnb[3] = (bnb[4] << 31) | (bnb[3] >> 1);
    bnb[4] = (bnb[5] << 31) | (bnb[4] >> 1);
    bnb[5] = (bnb[6] << 31) | (bnb[5] >> 1);
    bnb[6] = (bnb[7] << 31) | (bnb[6] >> 1);
    bnb[7] >>= 1;
}

/*
 * Unsigned comparison
 */
__device__ __forceinline__ int bn_ucmp_ge(const uint32 *a, const uint32 *b)
{
    int l = 0, g = 0;

#define bn_ucmp_ge_inner1_r(i)            \
        if (a[i] < b[i]) l |= (1 << i);   \
        if (a[i] > b[i]) g |= (1 << i);

    unroll_7_0(bn_ucmp_ge_inner1_r);

    return (l > g) ? 0 : 1;
}

/*
 * Negate
 */
__device__ __forceinline__ void bn_neg(uint32 *n)
{
    int c = 1;

    c = (n[0] = (~n[0]) + c) ? 0 : c;
    c = (n[1] = (~n[1]) + c) ? 0 : c;
    c = (n[2] = (~n[2]) + c) ? 0 : c;
    c = (n[3] = (~n[3]) + c) ? 0 : c;
    c = (n[4] = (~n[4]) + c) ? 0 : c;
    c = (n[5] = (~n[5]) + c) ? 0 : c;
    c = (n[6] = (~n[6]) + c) ? 0 : c;
    c = (n[7] = (~n[7]) + c) ? 0 : c;

}

#define bn_sub_word(r, a, b, t, c) do {     \
        t = a - b;                          \
        c = (a < b) ? 1 : 0;                \
        r = t;                              \
    } while (0)

#define bn_subb_word(r, a, b, t, c) do {    \
        t = a - (b + c);                    \
        c = (!(a) && c) ? 1 : 0;            \
        c |= (a < b) ? 1 : 0;               \
        r = t;                              \
    } while (0)

__device__ __forceinline__ uint32 bn_usub_words_seq(uint32 *r, uint32 *a, const uint32 *b)
{
    uint32 t, c = 0;

    bn_sub_word(r[0], a[0], b[0], t, c);

#define bn_usub_words_seq_inner1(i)         \
        bn_subb_word(r[i], a[i], b[i], t, c);

    unroll_1_7(bn_usub_words_seq_inner1);
    return c;
}

__device__ __forceinline__ void bn_mod_inverse(uint32 *R, uint32 *V)
{
    const uint32 modulus[8] = {0xffffffed, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x7fffffff};

    uint32 U[8], S[8];
    uint32 xc = 0, yc = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        U[i] = modulus[i];
        S[i] = 0;
        R[i] = 0;
    }
    __syncthreads();

    S[0] = 1;

    while (!bn_is_zero(V)) { // -------------
        while (!bn_is_odd(U)) { // -------------
            if (bn_is_odd(R)){
                yc += add_reduce_carry(R, R, modulus);
            }

            bn_rshift1_2(R, U);
            R[7] |= (yc << 31);
            yc >>= 1;
        }

        while (!bn_is_odd(V)) { // -------------
            if (bn_is_odd(S)){
                xc += add_reduce_carry(S, S, modulus);
            }

            bn_rshift1_2(S, V);
            S[7] |= (xc << 31);
            xc >>= 1;
        }

        if (!bn_ucmp_ge(V, U)) { // -------------
            sub_mod(U, U, V);
            yc += xc + add_reduce_carry(R, R, S);
        } else {
            sub_mod(V, V, U);
            xc += yc + add_reduce_carry(S, S, R);
        }
    }

    while (yc < 0x80000000){ // -------------
        yc -= bn_usub_words_seq(R, R, modulus);
    }
    __syncthreads();

    bn_neg(R);
}

// Computes Z = X*X mod P times count
__device__ __forceinline__ void sq_times(unsigned int* Y, const unsigned int* X, int count)
{
    unsigned int T[8];
    copy(T, X);

    do {
        mul_reduce_2(Y, T, T);
        copy(T, Y);
    } while (--count);
}

__device__ __forceinline__ void invert(unsigned int* Z){
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