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

