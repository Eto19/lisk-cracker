typedef unsigned long long int uint64;
typedef unsigned int uint32; 
typedef unsigned short uint16;
typedef unsigned char uint8;

typedef union 
{
    uint8 u8[4];
    uint32 u32[1];
} t32; 

typedef union 
{
    uint8 u8[8];
    uint32 u32[2];
    uint64 u64[1];
} t64; 

typedef union 
{
    uint8 u8[16];
    uint16 u16[8];
    uint32 u32[4];
    ulonglong2  u128;
} t128; 

typedef union
{
    uint8  u8[32];
    uint16 u16[16];
    uint32 u32[8];
    uint64 u64[4];
    ulonglong2  u128[2]; 
    ulonglong4  u256;
} t256;

typedef union
{
    uint16 u16[32];
    uint32 u32[16];
    uint64 u64[8];
    ulonglong2  u128[4];
    ulonglong4  u256[2];
} t512;

#define copy(a, b) do {\
    a[0] = b[0];\
    a[1] = b[1];\
    a[2] = b[2];\
    a[3] = b[3];\
    a[4] = b[4];\
    a[5] = b[5];\
    a[6] = b[6];\
    a[7] = b[7];\
}while(0)\

#define copy32x8(a, b) do {\
    a[0] = b[0];\
    a[1] = b[1];\
    a[2] = b[2];\
    a[3] = b[3];\
    a[4] = b[4];\
    a[5] = b[5];\
    a[6] = b[6];\
    a[7] = b[7];\
    a[8] = b[8];\
    a[9] = b[9];\
    a[10] = b[10];\
    a[11] = b[11];\
    a[12] = b[12];\
    a[13] = b[13];\
    a[14] = b[14];\
    a[15] = b[15];\
    a[16] = b[16];\
    a[17] = b[17];\
    a[18] = b[18];\
    a[19] = b[19];\
    a[20] = b[20];\
    a[21] = b[21];\
    a[22] = b[22];\
    a[23] = b[23];\
    a[24] = b[24];\
    a[25] = b[25];\
    a[26] = b[26];\
    a[27] = b[27];\
    a[28] = b[28];\
    a[29] = b[29];\
    a[30] = b[30];\
    a[31] = b[31];\
}while(0)\

#define copy64(a, b) do {\
    a[0] = b[0];\
    a[1] = b[1];\
    a[2] = b[2];\
    a[3] = b[3];\
}while(0)\

#define copy64_2(a, b) do {\
    a[0] = b[4];\
    a[1] = b[5];\
    a[2] = b[6];\
    a[3] = b[7];\
}while(0)\

/* loop unrolling */
#define unroll_1_0(a) do { a(1) a(0) } while (0)
#define unroll_3_0(a) do { a(3) a(2) a(1) a(0) } while (0)
#define unroll_4(a) do { a(0) a(1) a(2) a(3) } while (0)
#define unroll_8(a) do { a(0) a(1) a(2) a(3) a(4) a(5) a(6) a(7) } while (0)
#define unroll_1_7(a) do { a(1) a(2) a(3) a(4) a(5) a(6) a(7) } while (0)
#define unroll_7(a) do { a(0) a(1) a(2) a(3) a(4) a(5) a(6) } while (0)
#define unroll_7_0(a) do { a(7) a(6) a(5) a(4) a(3) a(2) a(1) a(0) } while (0)