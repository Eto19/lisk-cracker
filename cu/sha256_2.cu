#include "mul256.cu"

#ifndef BLOCKS
#define BLOCKS 3584*16
#endif

#ifndef N_BATCH
#define N_BATCH 256
#endif 

#ifndef SIZE_LIST
#define SIZE_LIST 500
#endif

#ifndef SIZE_LIST_ODD
#define SIZE_LIST_ODD 0
#endif

#ifndef BLOOM_FILTER_SIZE
#define BLOOM_FILTER_SIZE 71836
#endif

#ifndef K_HASH
#define K_HASH 3
#endif

#if SIZE_LIST_ODD
typedef union 
{ 
    uint64 u64[SIZE_LIST];
} haystack_t;
#else
typedef union 
{ 
    uint64 u64[SIZE_LIST];
    ulonglong2 u128[SIZE_LIST/2];
} haystack_t;
#endif

#if SIZE_LIST_ODD
typedef union 
{ 
    uint32 u32[SIZE_LIST];
    uint64 u64[SIZE_LIST/2];
} haystack2_t;
#else
typedef union 
{ 
    uint32 u32[SIZE_LIST];
    uint64 u64[SIZE_LIST/2];
    ulonglong2 u128[SIZE_LIST/4];
} haystack2_t;
#endif


// SHA256
__device__ __forceinline__ uint32 swap32_S(const uint32 x)
{
    uint32 r;
    asm("prmt.b32 %0, %1, 0, 0x0123;" : "=r"(r) : "r"(x));
    return r;
}

__device__ __constant__ uint32 K[] =
{
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
};

__device__ __constant__ uint32 I[] = {
  0x6a09e667ul,
  0xbb67ae85ul,
  0x3c6ef372ul,
  0xa54ff53aul,
  0x510e527ful,
  0x9b05688cul,
  0x1f83d9abul,
  0x5be0cd19ul,
};

//#define ROR(x,n) ((x>>n)|(x<<(32-n)))
__device__ __forceinline__ uint32 ROR(const uint32 x, const uint32 n) {

  uint32 y;
  asm("{\n\t"
    "shf.r.wrap.b32 %0, %1, %1, %2; \n\t"
    "}\n\t"
    : "=r"(y) : "r" (x), "r" (n));
  return y;
}

//#define S0(x) (ROR(x,2) ^ ROR(x,13) ^ ROR(x,22))
//#define S1(x) (ROR(x,6) ^ ROR(x,11) ^ ROR(x,25))

#define S0(x) (ROR(ROR(ROR(x,9) ^ x, 11) ^ x, 2))
#define S1(x) (ROR(ROR(ROR(x,14) ^ x, 5) ^ x, 6))

#define s0(x) (ROR(x,7) ^ ROR(x,18) ^ (x >> 3))
#define s1(x) (ROR(x,17) ^ ROR(x,19) ^ (x >> 10))

//#define Maj(x,y,z) ((x&y)^(x&z)^(y&z))
//#define Ch(x,y,z)  ((x&y)^(~x&z))

#define Maj(x,y,z) ((x & y) | (z & (x | y)))
#define Ch(x,y,z) (z ^ (x & (y ^ z)))

// SHA-256 inner round
#define S2Round(a, b, c, d, e, f, g, h, k, w) \
    t1 = h + S1(e) + Ch(e,f,g) + k + (w); \
    d += t1; \
    t2 = S0(a) + Maj(a,b,c); \
    h = t1 + t2;

#define S2Round_LAST(a, b, c, d, e, f, g, h, k, w) \
    t1 = h + S1(e) + Ch(e,f,g) + k + (w); \
    t2 = S0(a) + Maj(a,b,c); \
    h = t1 + t2;

// WMIX
#define WMIX() { \
w[0] += s1(w[14]) + w[9] + s0(w[1]);\
w[1] += s1(w[15]) + w[10] + s0(w[2]);\
w[2] += s1(w[0]) + w[11] + s0(w[3]);\
w[3] += s1(w[1]) + w[12] + s0(w[4]);\
w[4] += s1(w[2]) + w[13] + s0(w[5]);\
w[5] += s1(w[3]) + w[14] + s0(w[6]);\
w[6] += s1(w[4]) + w[15] + s0(w[7]);\
w[7] += s1(w[5]) + w[0] + s0(w[8]);\
w[8] += s1(w[6]) + w[1] + s0(w[9]);\
w[9] += s1(w[7]) + w[2] + s0(w[10]);\
w[10] += s1(w[8]) + w[3] + s0(w[11]);\
w[11] += s1(w[9]) + w[4] + s0(w[12]);\
w[12] += s1(w[10]) + w[5] + s0(w[13]);\
w[13] += s1(w[11]) + w[6] + s0(w[14]);\
w[14] += s1(w[12]) + w[7] + s0(w[15]);\
w[15] += s1(w[13]) + w[8] + s0(w[0]);\
}

#define SHA256_RND(k) {\
S2Round(a, b, c, d, e, f, g, h, K[k], w[0]);\
S2Round(h, a, b, c, d, e, f, g, K[k + 1], w[1]);\
S2Round(g, h, a, b, c, d, e, f, K[k + 2], w[2]);\
S2Round(f, g, h, a, b, c, d, e, K[k + 3], w[3]);\
S2Round(e, f, g, h, a, b, c, d, K[k + 4], w[4]);\
S2Round(d, e, f, g, h, a, b, c, K[k + 5], w[5]);\
S2Round(c, d, e, f, g, h, a, b, K[k + 6], w[6]);\
S2Round(b, c, d, e, f, g, h, a, K[k + 7], w[7]);\
S2Round(a, b, c, d, e, f, g, h, K[k + 8], w[8]);\
S2Round(h, a, b, c, d, e, f, g, K[k + 9], w[9]);\
S2Round(g, h, a, b, c, d, e, f, K[k + 10], w[10]);\
S2Round(f, g, h, a, b, c, d, e, K[k + 11], w[11]);\
S2Round(e, f, g, h, a, b, c, d, K[k + 12], w[12]);\
S2Round(d, e, f, g, h, a, b, c, K[k + 13], w[13]);\
S2Round(c, d, e, f, g, h, a, b, K[k + 14], w[14]);\
S2Round(b, c, d, e, f, g, h, a, K[k + 15], w[15]);\
}

#define SHA256_RND_LAST(k) {\
S2Round(a, b, c, d, e, f, g, h, K[k], w[0]);\
S2Round(h, a, b, c, d, e, f, g, K[k + 1], w[1]);\
S2Round(g, h, a, b, c, d, e, f, K[k + 2], w[2]);\
S2Round(f, g, h, a, b, c, d, e, K[k + 3], w[3]);\
S2Round(e, f, g, h, a, b, c, d, K[k + 4], w[4]);\
S2Round(d, e, f, g, h, a, b, c, K[k + 5], w[5]);\
S2Round(c, d, e, f, g, h, a, b, K[k + 6], w[6]);\
S2Round(b, c, d, e, f, g, h, a, K[k + 7], w[7]);\
S2Round(a, b, c, d, e, f, g, h, K[k + 8], w[8]);\
S2Round(h, a, b, c, d, e, f, g, K[k + 9], w[9]);\
S2Round(g, h, a, b, c, d, e, f, K[k + 10], w[10]);\
S2Round(f, g, h, a, b, c, d, e, K[k + 11], w[11]);\
S2Round(e, f, g, h, a, b, c, d, K[k + 12], w[12]);\
S2Round(d, e, f, g, h, a, b, c, K[k + 13], w[13]);\
S2Round(c, d, e, f, g, h, a, b, K[k + 14], w[14]);\
S2Round_LAST(b, c, d, e, f, g, h, a, K[k + 15], w[15]);\
}

// WMIX
#define WMIX_1() { \
w[0] += s0(w[1]);\
w[1] += 10485760 + s0(w[2]);\
w[2] += s1(w[0]) + s0(w[3]);\
w[3] += s1(w[1]) + s0(w[4]);\
w[4] += s1(w[2]) + s0(w[5]);\
w[5] += s1(w[3]) + s0(w[6]);\
temp_2[0] = w[0];\
temp_2[1] = w[1];\
temp_2[2] = w[2];\
temp_2[3] = w[3];\
temp_2[4] = w[4];\
temp_2[5] = w[5];\
w[6] += s1(w[4]) + w[15] + s0(w[7]);\
w[7] += s1(w[5]) + w[0] + 285220864;\
w[8] += s1(w[6]) + w[1];\
w[9] += s1(w[7]) + w[2];\
w[10] += s1(w[8]) + w[3];\
w[11] += s1(w[9]) + w[4];\
w[12] += s1(w[10]) + w[5];\
w[13] += s1(w[11]) + w[6];\
w[14] += s1(w[12]) + w[7] + 4194338;\
w[15] += s1(w[13]) + w[8] + s0(w[0]);\
}

#define SHA256_RND_1(k) {\
S2Round(a, b, c, d, e, f, g, h, K[k], w[0]);\
S2Round(h, a, b, c, d, e, f, g, K[k + 1], w[1]);\
S2Round(g, h, a, b, c, d, e, f, K[k + 2], w[2]);\
S2Round(f, g, h, a, b, c, d, e, K[k + 3], w[3]);\
S2Round(e, f, g, h, a, b, c, d, K[k + 4], w[4]);\
S2Round(d, e, f, g, h, a, b, c, K[k + 5], w[5]);\
S2Round(c, d, e, f, g, h, a, b, K[k + 6], w[6]);\
S2Round(b, c, d, e, f, g, h, a, K[k + 7], w[7]);\
temp_1[0] = a;\
temp_1[1] = b;\
temp_1[2] = c;\
temp_1[3] = d;\
temp_1[4] = e;\
temp_1[5] = f;\
temp_1[6] = g;\
temp_1[7] = h;\
S2Round(a, b, c, d, e, f, g, h, K[k + 8], 0x80000000);\
S2Round(h, a, b, c, d, e, f, g, K[k + 9], 0);\
S2Round(g, h, a, b, c, d, e, f, K[k + 10], 0);\
S2Round(f, g, h, a, b, c, d, e, K[k + 11], 0);\
S2Round(e, f, g, h, a, b, c, d, K[k + 12], 0);\
S2Round(d, e, f, g, h, a, b, c, K[k + 13], 0);\
S2Round(c, d, e, f, g, h, a, b, K[k + 14], 0);\
S2Round(b, c, d, e, f, g, h, a, K[k + 15], 256);\
}

#define DEF(x,y) uint32 x = I[y]

// Perform SHA-256 transformations, process 64-byte chunks
__device__ __forceinline__ void SHA256Transform(uint32 s[2], uint32* w, uint32* temp_1, uint32* temp_2) {
    uint32 t1;
    uint32 t2;

    DEF(a, 0);
    DEF(b, 1);
    DEF(c, 2);
    DEF(d, 3);
    DEF(e, 4);
    DEF(f, 5);
    DEF(g, 6);
    DEF(h, 7);

    SHA256_RND_1(0);
    WMIX_1();
    SHA256_RND(16);
    WMIX();
    SHA256_RND(32);
    WMIX();
    SHA256_RND_LAST(48);

    s[1] = swap32_S(a + I[0]);
    s[0] = swap32_S(b + I[1]);
}

// WMIX
#define WMIX_2() { \
w[6] += s1(w[4]) + w[15] + s0(w[7]);\
w[7] += s1(w[5]) + w[0] + 285220864;\
w[8] += s1(w[6]) + w[1];\
w[9] += s1(w[7]) + w[2];\
w[10] += s1(w[8]) + w[3];\
w[11] += s1(w[9]) + w[4];\
w[12] += s1(w[10]) + w[5];\
w[13] += s1(w[11]) + w[6];\
w[14] += s1(w[12]) + w[7] + 4194338;\
w[15] += s1(w[13]) + w[8] + s0(w[0]);\
}

#define SHA256_RND_2(k) {\
S2Round(a, b, c, d, e, f, g, h, K[k + 8], 0x80000000);\
S2Round(h, a, b, c, d, e, f, g, K[k + 9], 0);\
S2Round(g, h, a, b, c, d, e, f, K[k + 10], 0);\
S2Round(f, g, h, a, b, c, d, e, K[k + 11], 0);\
S2Round(e, f, g, h, a, b, c, d, K[k + 12], 0);\
S2Round(d, e, f, g, h, a, b, c, K[k + 13], 0);\
S2Round(c, d, e, f, g, h, a, b, K[k + 14], 0);\
S2Round(b, c, d, e, f, g, h, a, K[k + 15], 256);\
}

// Perform SHA-256 transformations, process 64-byte chunks
__device__ __forceinline__ void SHA256Transform_2(uint32 temp_1[8], uint32* w) {
    uint32 a, b, c, d, e, f, g, h;
    uint32 t1;
    uint32 t2;

    a = temp_1[0] + 128;
    b = temp_1[1];
    c = temp_1[2];
    d = temp_1[3];
    e = temp_1[4] + 128;
    f = temp_1[5];
    g = temp_1[6];
    h = temp_1[7];

    SHA256_RND_2(0);
    WMIX_2();
    SHA256_RND(16);
    WMIX();
    SHA256_RND(32);
    WMIX();
    SHA256_RND_LAST(48);

    temp_1[1] = swap32_S(a + I[0]);
    temp_1[0] = swap32_S(b + I[1]);
}

__device__ __forceinline__ int f_b_search(const uint64 needle, const uint64 *haystack)
{
    bool cmp;
    int mid, high, low;

    // Binary-search
    for (high = (SIZE_LIST-1), low = 0, mid = high / 2; high >= low ; mid = (high + low) / 2){
        cmp = (haystack[mid] < needle);

        low = !cmp ? low : (mid + 1);
        high = cmp ? high : (mid - 1);

        if (haystack[mid] == needle) {
            return mid;
        }
    }
    return -1;
}

__device__ __forceinline__ int f_b_search_2(const uint32 needle, const uint32 *haystack)
{
    bool cmp;
    int mid, high, low;

    // Binary-search
    for (high = (SIZE_LIST-1), low = 0, mid = high / 2; high >= low ; mid = (high + low) / 2){
        cmp = (haystack[mid] < needle);

        low = !cmp ? low : (mid + 1);
        high = cmp ? high : (mid - 1);

        if (haystack[mid] == needle) {
            return mid;
        }
    }
    return -1;
}

__device__ __forceinline__ uint32 hash1(const uint32 val) {
    uint32 x = val;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

__device__ __forceinline__ uint32 hash2(const uint32 val)
{
    uint32 x = val;
    x ^= x >> 15;
    x *= 0x2c1b3c6d;
    x ^= x >> 12;
    x *= 0x297a2d39;
    x ^= x >> 15;
    return x;
}

__device__ __forceinline__ uint32 bf_reduce(const uint32 hash) {
    return ((uint64)hash * (uint64)BLOOM_FILTER_SIZE) >> 32;
}

__device__ __forceinline__ bool bloom_filter_check(const uint32 value1, const uint32 value2, const uint8 *bloom_filter){
    uint32 key = 0;
    uint32 h1 = hash1(value1);
    uint32 h2 = hash2(value1);

#pragma unroll
    for (uint32 i = 1; i < K_HASH+1; i++)
    {
        key = bf_reduce(h1+i*h2);
        if(!((bloom_filter[key>>3] >> (key&7)) & 1)){
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ void bloom_filter_update(const uint32 value1, const uint32 value2, uint8 *bloom_filter){
    uint32 key = 0;
    uint32 h1 = hash1(value1);
    uint32 h2 = hash2(value1);

#pragma unroll
    for (int i = 1; i < K_HASH+1; i++)
    {
        key = bf_reduce(h1+i*h2);
        bloom_filter[key>>3] |= (1 << (key&7));
    }
}

__global__ void bloom_filter_init(const uint32 *haystack, const uint32 *haystack2, uint8 *bloom_filter)
{
    for (int i = 0; i < SIZE_LIST; i++)
    {
        bloom_filter_update(haystack[i], haystack2[i], &bloom_filter[0]);
    }
}


__global__ void sha256_quad(const t256 *in, uint32 *temp, const t256 *inv_z, const uint8 *bloom_filter, const haystack2_t *haystack, const haystack2_t *haystack2, const haystack_t *full_haystack, int *found)
{
    t256 xy, z_in;
    uint32 publicKeyBytes[16], s[2], temp_1[8], temp_2[6];

    const int gid = blockDim.x * blockIdx.x + threadIdx.x;

    const int cid_1 = ((gid>>1) < (BLOCKS * N_BATCH));
    const int cid_2 = ((gid>>1) < (BLOCKS * N_BATCH)*2);
    const int cid_3 = ((gid>>1) < (BLOCKS * N_BATCH)*3);

    const int c_id1 = (gid>>1) - (BLOCKS * N_BATCH);
    const int c_id2 = (gid>>1) - (BLOCKS * N_BATCH)*2;
    const int c_id3 = (gid>>1) - (BLOCKS * N_BATCH)*3;

    const int c_id = cid_1 ? gid>>1 : cid_2 ? c_id1 : cid_3 ? c_id2 : c_id3;

    // GLOBAL TO REGISTER TRANSFER
    copy(xy.u32, in[gid>>1].u32);
    copy(z_in.u32, inv_z[c_id].u32);

    mul_mod(xy.u32, z_in.u32, xy.u32);

    xy.u32[0] = (gid&1) ? (0xffffffed - xy.u32[0]) : xy.u32[0];
    xy.u32[1] = (gid&1) ? (0xffffffff - xy.u32[1]) : xy.u32[1];
    xy.u32[2] = (gid&1) ? (0xffffffff - xy.u32[2]) : xy.u32[2];
    xy.u32[3] = (gid&1) ? (0xffffffff - xy.u32[3]) : xy.u32[3];

    xy.u32[4] = (gid&1) ? (0xffffffff - xy.u32[4]) : xy.u32[4];
    xy.u32[5] = (gid&1) ? (0xffffffff - xy.u32[5]) : xy.u32[5];
    xy.u32[6] = (gid&1) ? (0xffffffff - xy.u32[6]) : xy.u32[6];
    xy.u32[7] = (gid&1) ? (0x7fffffff - xy.u32[7]) : xy.u32[7];

    #pragma unroll
    for(int x = 0; x < 8; x++){
        xy.u32[x] = swap32_S(xy.u32[x]);
    }

    publicKeyBytes[0] =  xy.u32[0];
    publicKeyBytes[1] =  xy.u32[1];
    publicKeyBytes[2] =  xy.u32[2];
    publicKeyBytes[3] =  xy.u32[3];
    publicKeyBytes[4] =  xy.u32[4];
    publicKeyBytes[5] =  xy.u32[5];
    publicKeyBytes[6] =  xy.u32[6];
    publicKeyBytes[7] =  xy.u32[7];
    publicKeyBytes[8] = 0x80000000;
    publicKeyBytes[9] = 0;
    publicKeyBytes[10] = 0;
    publicKeyBytes[11] = 0;
    publicKeyBytes[12] = 0;
    publicKeyBytes[13] = 0;
    publicKeyBytes[14] = 0;
    publicKeyBytes[15] = 256;

    SHA256Transform(s, publicKeyBytes, temp_1, temp_2);

    int test_1;

    t64 full_address;
    full_address.u32[0] = s[1];
    full_address.u32[1] = s[0];

    if(bloom_filter_check(s[0], s[1], &bloom_filter[0]))
    {
        test_1 = f_b_search_2(s[0], &haystack2->u32[0]);
        if (test_1 != -1)
        {
            test_1 = f_b_search(full_address.u64[0], &full_haystack->u64[0]);
            if (test_1 != -1)
            {
                *found = gid;
            }
        }
    }

    publicKeyBytes[0] = temp_2[0];
    publicKeyBytes[1] = temp_2[1];
    publicKeyBytes[2] = temp_2[2];
    publicKeyBytes[3] = temp_2[3];
    publicKeyBytes[4] = temp_2[4];
    publicKeyBytes[5] = temp_2[5];
    publicKeyBytes[6] = xy.u32[6];
    publicKeyBytes[7] = xy.u32[7]^128;
    publicKeyBytes[8] = 0x80000000;
    publicKeyBytes[9] = 0;
    publicKeyBytes[10] = 0;
    publicKeyBytes[11] = 0;
    publicKeyBytes[12] = 0;
    publicKeyBytes[13] = 0;
    publicKeyBytes[14] = 0;
    publicKeyBytes[15] = 256;

    SHA256Transform_2(temp_1, publicKeyBytes);

    full_address.u32[0] = temp_1[1];
    full_address.u32[1] = temp_1[0];

    if (bloom_filter_check(temp_1[0], temp_1[1], &bloom_filter[0]))
    {
        test_1 = f_b_search_2(temp_1[0], &haystack2->u32[0]);
        if (test_1 != -1)
        {
            test_1 = f_b_search(full_address.u64[0], &full_haystack->u64[0]);
            if (test_1 != -1)
            {
                *found = gid;
            }
        }
    }
}