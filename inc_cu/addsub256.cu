#define bn_is_odd(bn) (bn[0] & 1)
#define bn_is_zero(bn) (!bn[0] && !bn[1] && !bn[2] && !bn[3] && !bn[4] && !bn[5] && !bn[6] && !bn[7])

__device__ __forceinline__ void mod_p(unsigned int *X)
{
    asm("{\n\t"
      ".reg .u32 r0;\n\t"
      "sub.cc.u32      %0, %8, 0xffffffed; \n\t"
      "subc.cc.u32     %1, %9, 0xffffffff; \n\t"
      "subc.cc.u32     %2, %10, 0xffffffff; \n\t"
      "subc.cc.u32     %3, %11, 0xffffffff; \n\t"
      "subc.cc.u32     %4, %12, 0xffffffff; \n\t"
      "subc.cc.u32     %5, %13, 0xffffffff; \n\t"
      "subc.cc.u32     %6, %14, 0xffffffff; \n\t"
      "subc.cc.u32     %7, %15, 0x7fffffff; \n\t"
      "subc.u32        r0, 0, 0; \n\t"

      "and.b32         %8, r0, 0xffffffed; \n\t"
      "add.cc.u32      %0, %0, %8; \n\t"
      "addc.cc.u32     %1, %1, r0; \n\t"
      "addc.cc.u32     %2, %2, r0; \n\t"
      "addc.cc.u32     %3, %3, r0; \n\t"
      "addc.cc.u32     %4, %4, r0; \n\t"
      "addc.cc.u32     %5, %5, r0; \n\t"
      "addc.cc.u32     %6, %6, r0; \n\t"
      "shr.u32         %8, r0,  1; \n\t"
      "addc.u32        %7, %7, %8; \n\t"

      "sub.cc.u32      %0, %0, 0xffffffed; \n\t"
      "subc.cc.u32     %1, %1, 0xffffffff; \n\t"
      "subc.cc.u32     %2, %2, 0xffffffff; \n\t"
      "subc.cc.u32     %3, %3, 0xffffffff; \n\t"
      "subc.cc.u32     %4, %4, 0xffffffff; \n\t"
      "subc.cc.u32     %5, %5, 0xffffffff; \n\t"
      "subc.cc.u32     %6, %6, 0xffffffff; \n\t"
      "subc.cc.u32     %7, %7, 0x7fffffff; \n\t"
      "subc.u32        r0, 0, 0; \n\t"

      "and.b32         %8, r0, 0xffffffed; \n\t"
      "add.cc.u32      %0, %0, %8; \n\t"
      "addc.cc.u32     %1, %1, r0; \n\t"
      "addc.cc.u32     %2, %2, r0; \n\t"
      "addc.cc.u32     %3, %3, r0; \n\t"
      "addc.cc.u32     %4, %4, r0; \n\t"
      "addc.cc.u32     %5, %5, r0; \n\t"
      "addc.cc.u32     %6, %6, r0; \n\t"
      "shr.u32         %8, r0,  1; \n\t"
      "addc.u32        %7, %7, %8; \n\t"
      "}"
      : "=r"(X[0]), "=r"(X[1]), "=r"(X[2]), "=r"(X[3]),
        "=r"(X[4]), "=r"(X[5]), "=r"(X[6]), "=r"(X[7])
      : "r"(X[0]), "r"(X[1]), "r"(X[2]), "r"(X[3]), "r"(X[4]), "r"(X[5]),
        "r"(X[6]), "r"(X[7]));
}

/* Z = X+Y mod P */
__device__ __forceinline__ void add_reduce(uint32* Z, const uint32* X, const uint32* Y)
{
    asm("{\n\t"
      ".reg .u32 r0;\n\t"
      "add.cc.u32      %0, %8, %16; \n\t"
      "addc.cc.u32     %1, %9, %17; \n\t"
      "addc.cc.u32     %2, %10, %18; \n\t"
      "addc.cc.u32     %3, %11, %19; \n\t"
      "addc.cc.u32     %4, %12, %20; \n\t"
      "addc.cc.u32     %5, %13, %21; \n\t"
      "addc.cc.u32     %6, %14, %22; \n\t"
      "addc.cc.u32     %7, %15, %23; \n\t"
      "addc.u32        r0,  0,  0; \n\t"
      "mul.lo.u32      r0, r0, 38; \n\t"
      "add.cc.u32      %0, %0, r0; \n\t"
      "addc.u32        %1, %1,  0; \n\t"
      "}"
      : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3]),
        "=r"(Z[4]), "=r"(Z[5]), "=r"(Z[6]), "=r"(Z[7])
      : "r"(X[0]), "r"(X[1]), "r"(X[2]), "r"(X[3]), "r"(X[4]), "r"(X[5]),
        "r"(X[6]), "r"(X[7]), "r"(Y[0]), "r"(Y[1]), "r"(Y[2]), "r"(Y[3]),
        "r"(Y[4]), "r"(Y[5]), "r"(Y[6]), "r"(Y[7]));
}

/* Z = X-Y mod P */
__device__ __forceinline__ void sub_reduce(uint32* Z, const uint32* X, const uint32* Y)
{
    asm("{\n\t"
      ".reg .u32 r0;\n\t"
      "sub.cc.u32      %0, %8, %16; \n\t"
      "subc.cc.u32     %1, %9, %17; \n\t"
      "subc.cc.u32     %2, %10, %18; \n\t"
      "subc.cc.u32     %3, %11, %19; \n\t"
      "subc.cc.u32     %4, %12, %20; \n\t"
      "subc.cc.u32     %5, %13, %21; \n\t"
      "subc.cc.u32     %6, %14, %22; \n\t"
      "subc.cc.u32     %7, %15, %23; \n\t"
      "subc.u32        r0,  0,  0; \n\t"
      "and.b32         r0, r0, 38; \n\t"
      "sub.cc.u32      %0, %0, r0; \n\t"
      "subc.cc.u32     %1, %1,  0; \n\t"
      "subc.cc.u32     %2, %2,  0; \n\t"
      "subc.cc.u32     %3, %3,  0; \n\t"
      "subc.cc.u32     %4, %4,  0; \n\t"
      "subc.cc.u32     %5, %5,  0; \n\t"
      "subc.cc.u32     %6, %6,  0; \n\t"
      "subc.u32        %7, %7,  0; \n\t"
      "}"
      : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3]),
        "=r"(Z[4]), "=r"(Z[5]), "=r"(Z[6]), "=r"(Z[7])
      : "r"(X[0]), "r"(X[1]), "r"(X[2]), "r"(X[3]), "r"(X[4]), "r"(X[5]),
        "r"(X[6]), "r"(X[7]), "r"(Y[0]), "r"(Y[1]), "r"(Y[2]), "r"(Y[3]),
        "r"(Y[4]), "r"(Y[5]), "r"(Y[6]), "r"(Y[7]));
}

/* Z = X+Y mod P */
__device__ __forceinline__ void add_mod(uint32* Z, const uint32* X, const uint32* Y)
{
    asm("{\n\t"
      ".reg .u32 r0;\n\t"
      "add.cc.u32      %0, %8, %16; \n\t"
      "addc.cc.u32     %1, %9, %17; \n\t"
      "addc.cc.u32     %2, %10, %18; \n\t"
      "addc.cc.u32     %3, %11, %19; \n\t"
      "addc.cc.u32     %4, %12, %20; \n\t"
      "addc.cc.u32     %5, %13, %21; \n\t"
      "addc.cc.u32     %6, %14, %22; \n\t"
      "addc.cc.u32     %7, %15, %23; \n\t"
      "addc.u32        r0,  0,  0; \n\t"
      "mul.lo.u32      r0, r0, 38; \n\t"
      "add.cc.u32      %0, %0, r0; \n\t"
      "addc.u32        %1, %1,  0; \n\t"
      "}"
      : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3]),
        "=r"(Z[4]), "=r"(Z[5]), "=r"(Z[6]), "=r"(Z[7])
      : "r"(X[0]), "r"(X[1]), "r"(X[2]), "r"(X[3]), "r"(X[4]), "r"(X[5]),
        "r"(X[6]), "r"(X[7]), "r"(Y[0]), "r"(Y[1]), "r"(Y[2]), "r"(Y[3]),
        "r"(Y[4]), "r"(Y[5]), "r"(Y[6]), "r"(Y[7]));

    mod_p(Z);
}

/* Z = X-Y mod P */
__device__ __forceinline__ void sub_mod(uint32* Z, const uint32* X, const uint32* Y)
{
    asm("{\n\t"
      ".reg .u32 r0;\n\t"
      "sub.cc.u32      %0, %8, %16; \n\t"
      "subc.cc.u32     %1, %9, %17; \n\t"
      "subc.cc.u32     %2, %10, %18; \n\t"
      "subc.cc.u32     %3, %11, %19; \n\t"
      "subc.cc.u32     %4, %12, %20; \n\t"
      "subc.cc.u32     %5, %13, %21; \n\t"
      "subc.cc.u32     %6, %14, %22; \n\t"
      "subc.cc.u32     %7, %15, %23; \n\t"
      "subc.u32        r0,  0,  0; \n\t"
      "and.b32         r0, r0, 38; \n\t"
      "sub.cc.u32      %0, %0, r0; \n\t"
      "subc.cc.u32     %1, %1,  0; \n\t"
      "subc.cc.u32     %2, %2,  0; \n\t"
      "subc.cc.u32     %3, %3,  0; \n\t"
      "subc.cc.u32     %4, %4,  0; \n\t"
      "subc.cc.u32     %5, %5,  0; \n\t"
      "subc.cc.u32     %6, %6,  0; \n\t"
      "subc.u32        %7, %7,  0; \n\t"
      "}"
      : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3]),
        "=r"(Z[4]), "=r"(Z[5]), "=r"(Z[6]), "=r"(Z[7])
      : "r"(X[0]), "r"(X[1]), "r"(X[2]), "r"(X[3]), "r"(X[4]), "r"(X[5]),
        "r"(X[6]), "r"(X[7]), "r"(Y[0]), "r"(Y[1]), "r"(Y[2]), "r"(Y[3]),
        "r"(Y[4]), "r"(Y[5]), "r"(Y[6]), "r"(Y[7]));

    mod_p(Z);
}


/* Z = X+Y mod P */
__device__ __forceinline__ uint32 add_reduce_carry(uint32* Z, const uint32* X, const uint32* Y)
{
    uint32 carry = 0;

    asm("{\n\t"
      "add.cc.u32      %0, %9, %17; \n\t"
      "addc.cc.u32     %1, %10, %18; \n\t"
      "addc.cc.u32     %2, %11, %19; \n\t"
      "addc.cc.u32     %3, %12, %20; \n\t"
      "addc.cc.u32     %4, %13, %21; \n\t"
      "addc.cc.u32     %5, %14, %22; \n\t"
      "addc.cc.u32     %6, %15, %23; \n\t"
      "addc.cc.u32     %7, %16, %24; \n\t"
      "addc.u32        %8,  0,  0; \n\t"
      "}"
      : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3]),
        "=r"(Z[4]), "=r"(Z[5]), "=r"(Z[6]), "=r"(Z[7]), "=r"(carry)
      : "r"(X[0]), "r"(X[1]), "r"(X[2]), "r"(X[3]), "r"(X[4]), "r"(X[5]),
        "r"(X[6]), "r"(X[7]), "r"(Y[0]), "r"(Y[1]), "r"(Y[2]), "r"(Y[3]),
        "r"(Y[4]), "r"(Y[5]), "r"(Y[6]), "r"(Y[7]));

    return carry;
}

/* Z = X-Y mod P */
__device__ __forceinline__ uint32 sub_reduce_carry(uint32* Z, const uint32* X, const uint32* Y)
{
    uint32 carry = 0;

    asm("{\n\t"
      "sub.cc.u32      %0, %9, %17; \n\t"
      "subc.cc.u32     %1, %10, %18; \n\t"
      "subc.cc.u32     %2, %11, %19; \n\t"
      "subc.cc.u32     %3, %12, %20; \n\t"
      "subc.cc.u32     %4, %13, %21; \n\t"
      "subc.cc.u32     %5, %14, %22; \n\t"
      "subc.cc.u32     %6, %15, %23; \n\t"
      "subc.cc.u32     %7, %16, %24; \n\t"
      "addc.u32        %8,  0,  0; \n\t"
      "}"
      : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3]),
        "=r"(Z[4]), "=r"(Z[5]), "=r"(Z[6]), "=r"(Z[7]), "=r"(carry)
      : "r"(X[0]), "r"(X[1]), "r"(X[2]), "r"(X[3]), "r"(X[4]), "r"(X[5]),
        "r"(X[6]), "r"(X[7]), "r"(Y[0]), "r"(Y[1]), "r"(Y[2]), "r"(Y[3]),
        "r"(Y[4]), "r"(Y[5]), "r"(Y[6]), "r"(Y[7]));

    return carry;
}

/* Z = X-Y mod P */
__device__ __forceinline__ void sub_4(uint32* Z, const uint32* Y)
{
    asm("{\n\t"
      ".reg .u32 r0;\n\t"
      "sub.cc.u32      %0, 4, %8; \n\t"
      "subc.cc.u32     %1, 0, %9; \n\t"
      "subc.cc.u32     %2, 0, %10; \n\t"
      "subc.cc.u32     %3, 0, %11; \n\t"
      "subc.cc.u32     %4, 0, %12; \n\t"
      "subc.cc.u32     %5, 0, %13; \n\t"
      "subc.cc.u32     %6, 0, %14; \n\t"
      "subc.cc.u32     %7, 0, %15; \n\t"
      "subc.u32        r0,  0,  0; \n\t"
      "and.b32         r0, r0, 38; \n\t"
      "sub.cc.u32      %0, %0, r0; \n\t"
      "subc.u32        %1, %1,  0; \n\t"
      "}"
      : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3]),
        "=r"(Z[4]), "=r"(Z[5]), "=r"(Z[6]), "=r"(Z[7])
      : "r"(Y[0]), "r"(Y[1]), "r"(Y[2]), "r"(Y[3]),
        "r"(Y[4]), "r"(Y[5]), "r"(Y[6]), "r"(Y[7]));
}

/* Z = X-Y mod P */
__device__ __forceinline__ void sub_2(uint32* Z, const uint32* Y)
{
    asm("{\n\t"
      ".reg .u32 r0;\n\t"
      "sub.cc.u32      %0, 2, %8; \n\t"
      "subc.cc.u32     %1, 0, %9; \n\t"
      "subc.cc.u32     %2, 0, %10; \n\t"
      "subc.cc.u32     %3, 0, %11; \n\t"
      "subc.cc.u32     %4, 0, %12; \n\t"
      "subc.cc.u32     %5, 0, %13; \n\t"
      "subc.cc.u32     %6, 0, %14; \n\t"
      "subc.cc.u32     %7, 0, %15; \n\t"
      "subc.u32        r0,  0,  0; \n\t"
      "and.b32         r0, r0, 38; \n\t"
      "sub.cc.u32      %0, %0, r0; \n\t"
      "subc.u32        %1, %1,  0; \n\t"
      "}"
      : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3]),
        "=r"(Z[4]), "=r"(Z[5]), "=r"(Z[6]), "=r"(Z[7])
      : "r"(Y[0]), "r"(Y[1]), "r"(Y[2]), "r"(Y[3]),
        "r"(Y[4]), "r"(Y[5]), "r"(Y[6]), "r"(Y[7]));
}

/* Z = X-Y mod P */
__device__ __forceinline__ void add_2(uint32* Z, const uint32* Y)
{
    asm("{\n\t"
      ".reg .u32 r0;\n\t"
      "add.cc.u32      %0, 2, %8; \n\t"
      "addc.cc.u32     %1, 0, %9; \n\t"
      "addc.cc.u32     %2, 0, %10; \n\t"
      "addc.cc.u32     %3, 0, %11; \n\t"
      "addc.cc.u32     %4, 0, %12; \n\t"
      "addc.cc.u32     %5, 0, %13; \n\t"
      "addc.cc.u32     %6, 0, %14; \n\t"
      "addc.cc.u32     %7, 0, %15; \n\t"
      "addc.u32        r0,  0,  0; \n\t"
      "mul.lo.u32      r0, r0, 38; \n\t"
      "add.cc.u32      %0, %0, r0; \n\t"
      "addc.u32        %1, %1,  0; \n\t"
      "}"
      : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3]),
        "=r"(Z[4]), "=r"(Z[5]), "=r"(Z[6]), "=r"(Z[7])
      : "r"(Y[0]), "r"(Y[1]), "r"(Y[2]), "r"(Y[3]),
        "r"(Y[4]), "r"(Y[5]), "r"(Y[6]), "r"(Y[7]));
}

/* Z = X-Y mod P */
__device__ __forceinline__ void sub_reduce128(uint32* Z, const uint32* X, const uint32* Y)
{
    asm("{\n\t"
      "sub.cc.u32      %0, %4, %8; \n\t"
      "subc.cc.u32     %1, %5, %9; \n\t"
      "subc.cc.u32     %2, %6, %10; \n\t"
      "subc.u32        %3, %7, %11; \n\t"
      "}"
      : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3])
      : "r"(X[0]), "r"(X[1]), "r"(X[2]), "r"(X[3]),
        "r"(Y[0]), "r"(Y[1]), "r"(Y[2]), "r"(Y[3])
        );
}

/* Z = ~Y */
__device__ __forceinline__ void neg256(uint32* Z, const uint32* Y)
{
    asm("{\n\t"
      ".reg .u32 r0;\n\t"
      "sub.cc.u32      %0, 0, %8; \n\t"
      "subc.cc.u32     %1, 0, %9; \n\t"
      "subc.cc.u32     %2, 0, %10; \n\t"
      "subc.cc.u32     %3, 0, %11; \n\t"
      "subc.cc.u32     %4, 0, %12; \n\t"
      "subc.cc.u32     %5, 0, %13; \n\t"
      "subc.cc.u32     %6, 0, %14; \n\t"
      "subc.cc.u32     %7, 0, %15; \n\t"
      "subc.u32        r0,  0,  0; \n\t"
      "and.b32         r0, r0, 38; \n\t"
      "sub.cc.u32      %0, %0, r0; \n\t"
      "subc.cc.u32     %1, %1,  0; \n\t"
      "subc.cc.u32     %2, %2,  0; \n\t"
      "subc.cc.u32     %3, %3,  0; \n\t"
      "subc.cc.u32     %4, %4,  0; \n\t"
      "subc.cc.u32     %5, %5,  0; \n\t"
      "subc.cc.u32     %6, %6,  0; \n\t"
      "subc.u32        %7, %7,  0; \n\t"
      "}"
      : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3]),
        "=r"(Z[4]), "=r"(Z[5]), "=r"(Z[6]), "=r"(Z[7])
      : "r"(Y[0]), "r"(Y[1]), "r"(Y[2]), "r"(Y[3]),
        "r"(Y[4]), "r"(Y[5]), "r"(Y[6]), "r"(Y[7]));
}

/* Z = ~Y */
__device__ __forceinline__ void neg256_mod(uint32* Z, const uint32* Y)
{
    asm("{\n\t"
      ".reg .u32 r0;\n\t"
      "sub.cc.u32      %0, 0xffffffed, %8; \n\t"
      "subc.cc.u32     %1, 0xffffffff, %9; \n\t"
      "subc.cc.u32     %2, 0xffffffff, %10; \n\t"
      "subc.cc.u32     %3, 0xffffffff, %11; \n\t"
      "subc.cc.u32     %4, 0xffffffff, %12; \n\t"
      "subc.cc.u32     %5, 0xffffffff, %13; \n\t"
      "subc.cc.u32     %6, 0xffffffff, %14; \n\t"
      "subc.cc.u32     %7, 0x7fffffff, %15; \n\t"
      "subc.u32        r0,  0,  0; \n\t"
      "and.b32         r0, r0, 38; \n\t"
      "sub.cc.u32      %0, %0, r0; \n\t"
      "subc.cc.u32     %1, %1,  0; \n\t"
      "subc.cc.u32     %2, %2,  0; \n\t"
      "subc.cc.u32     %3, %3,  0; \n\t"
      "subc.cc.u32     %4, %4,  0; \n\t"
      "subc.cc.u32     %5, %5,  0; \n\t"
      "subc.cc.u32     %6, %6,  0; \n\t"
      "subc.u32        %7, %7,  0; \n\t"
      "}"
      : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3]),
        "=r"(Z[4]), "=r"(Z[5]), "=r"(Z[6]), "=r"(Z[7])
      : "r"(Y[0]), "r"(Y[1]), "r"(Y[2]), "r"(Y[3]),
        "r"(Y[4]), "r"(Y[5]), "r"(Y[6]), "r"(Y[7]));

    mod_p(Z);
}

__device__ __forceinline__ void add_reduce_1(uint32* Z, const uint32* X, const uint32* Y)
{
    asm("{\n\t"
      "mov.b32         %2, %14;\n\t"
      "mov.b32         %3, %15;\n\t"
      "add.cc.u32      %4, %8, %16; \n\t"
      "addc.cc.u32     %5, %9, %17; \n\t"
      "addc.cc.u32     %6, %10, %18; \n\t"
      "addc.cc.u32     %7, %11, %19; \n\t"
      "addc.u32        %0,  0,  0; \n\t"
      "mul.lo.u32      %0, %0, 38; \n\t"
      "add.cc.u32      %0, %12, %0; \n\t"
      "addc.u32        %1, %13,  0; \n\t"
      "}"
      : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3]),
        "=r"(Z[4]), "=r"(Z[5]), "=r"(Z[6]), "=r"(Z[7])
      : "r"(X[0]), "r"(X[1]), "r"(X[2]), "r"(X[3]), 
        "r"(Y[0]), "r"(Y[1]), "r"(Y[2]), "r"(Y[3]),
        "r"(Y[4]), "r"(Y[5]), "r"(Y[6]), "r"(Y[7]));
}

__device__ __forceinline__ void add_reduce_2(uint32* Z, const uint32* X, const uint32* Y)
{
    asm("{\n\t"
      "add.cc.u32      %0, %5 ,  %9; \n\t"
      "addc.cc.u32     %1, %6 , %10; \n\t"
      "addc.cc.u32     %2, %7 , %11; \n\t"
      "addc.cc.u32     %3, %8 , %12; \n\t"
      "addc.u32        %4, %13,   0; \n\t"
      "}"
      : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3]),
        "=r"(Z[4])
      : "r"(X[0]), "r"(X[1]), "r"(X[2]), "r"(X[3]), 
        "r"(Y[0]), "r"(Y[1]), "r"(Y[2]), "r"(Y[3]),
        "r"(Y[4]));
}
