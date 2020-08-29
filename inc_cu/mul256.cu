/*
mul_reduce
mul_mod
mod_p
umul256wide

optimizations from :
https://forums.developer.nvidia.com/t/long-integer-multiplication-mul-wide-u64-and-mul-wide-u128/51520
*/

#include "util.cu"
#include "addsub256.cu"

// multiply two unsigned 256-bit integers into an unsigned 512-bit product
__device__ __forceinline__ void umul256wide(uint32 *x, uint32 *y)
{
    t512 res;
    t256 a;
    t256 b;

    copy(a.u32, x);
    copy(b.u32, y);

    asm("{\n\t"
        ".reg .u32 s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15;\n\t"
        ".reg .u32 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;\n\t"
        // Compute first partial sum
        "mul.wide.u16 %0, %16, %32;\n\t"
        "mul.wide.u16 %1, %16, %34;\n\t"
        "mul.wide.u16 %2, %16, %36;\n\t"
        "mul.wide.u16 %3, %16, %38;\n\t"
        "mul.wide.u16 %4, %16, %40;\n\t"
        "mul.wide.u16 %5, %16, %42;\n\t"
        "mul.wide.u16 %6, %16, %44;\n\t"
        "mul.wide.u16 %7, %16, %46;\n\t"
        "mul.wide.u16 %8, %31, %33;\n\t"
        "mul.wide.u16 %9, %31, %35;\n\t"
        "mul.wide.u16 %10, %31, %37;\n\t"
        "mul.wide.u16 %11, %31, %39;\n\t"
        "mul.wide.u16 %12, %31, %41;\n\t"
        "mul.wide.u16 %13, %31, %43;\n\t"
        "mul.wide.u16 %14, %31, %45;\n\t"
        "mul.wide.u16 %15, %31, %47;\n\t"
        "mul.wide.u16 t7, %17, %45;\n\t"
        "mul.wide.u16 t8, %30, %34;\n\t"
        "add.cc.u32 %7, %7, t7;\n\t"
        "addc.cc.u32 %8, %8, t8;\n\t"
        "addc.u32 %9, %9, 0;\n\t"
        "mul.wide.u16 t7, %18, %44;\n\t"
        "mul.wide.u16 t8, %29, %35;\n\t"
        "add.cc.u32 %7, %7, t7;\n\t"
        "addc.cc.u32 %8, %8, t8;\n\t"
        "addc.u32 %9, %9, 0;\n\t"
        "mul.wide.u16 t6, %17, %43;\n\t"
        "mul.wide.u16 t7, %19, %43;\n\t"
        "mul.wide.u16 t8, %28, %36;\n\t"
        "mul.wide.u16 t9, %30, %36;\n\t"
        "add.cc.u32 %6, %6, t6;\n\t"
        "addc.cc.u32 %7, %7, t7;\n\t"
        "addc.cc.u32 %8, %8, t8;\n\t"
        "addc.cc.u32 %9, %9, t9;\n\t"
        "addc.u32 %10, %10, 0;\n\t"
        "mul.wide.u16 t6, %18, %42;\n\t"
        "mul.wide.u16 t7, %20, %42;\n\t"
        "mul.wide.u16 t8, %27, %37;\n\t"
        "mul.wide.u16 t9, %29, %37;\n\t"
        "add.cc.u32 %6, %6, t6;\n\t"
        "addc.cc.u32 %7, %7, t7;\n\t"
        "addc.cc.u32 %8, %8, t8;\n\t"
        "addc.cc.u32 %9, %9, t9;\n\t"
        "addc.u32 %10, %10, 0;\n\t"
        "mul.wide.u16 t5, %17, %41;\n\t"
        "mul.wide.u16 t6, %19, %41;\n\t"
        "mul.wide.u16 t7, %21, %41;\n\t"
        "mul.wide.u16 t8, %26, %38;\n\t"
        "mul.wide.u16 t9, %28, %38;\n\t"
        "mul.wide.u16 t10, %30, %38;\n\t"
        "add.cc.u32 %5, %5, t5;\n\t"
        "addc.cc.u32 %6, %6, t6;\n\t"
        "addc.cc.u32 %7, %7, t7;\n\t"
        "addc.cc.u32 %8, %8, t8;\n\t"
        "addc.cc.u32 %9, %9, t9;\n\t"
        "addc.cc.u32 %10, %10, t10;\n\t"
        "addc.u32 %11, %11, 0;\n\t"
        "mul.wide.u16 t5, %18, %40;\n\t"
        "mul.wide.u16 t6, %20, %40;\n\t"
        "mul.wide.u16 t7, %22, %40;\n\t"
        "mul.wide.u16 t8, %25, %39;\n\t"
        "mul.wide.u16 t9, %27, %39;\n\t"
        "mul.wide.u16 t10, %29, %39;\n\t"
        "add.cc.u32 %5, %5, t5;\n\t"
        "addc.cc.u32 %6, %6, t6;\n\t"
        "addc.cc.u32 %7, %7, t7;\n\t"
        "addc.cc.u32 %8, %8, t8;\n\t"
        "addc.cc.u32 %9, %9, t9;\n\t"
        "addc.cc.u32 %10, %10, t10;\n\t"
        "addc.u32 %11, %11, 0;\n\t"
        "mul.wide.u16 t4, %17, %39;\n\t"
        "mul.wide.u16 t5, %19, %39;\n\t"
        "mul.wide.u16 t6, %21, %39;\n\t"
        "mul.wide.u16 t7, %23, %39;\n\t"
        "mul.wide.u16 t8, %24, %40;\n\t"
        "mul.wide.u16 t9, %26, %40;\n\t"
        "mul.wide.u16 t10, %28, %40;\n\t"
        "mul.wide.u16 t11, %30, %40;\n\t"
        "add.cc.u32 %4, %4, t4;\n\t"
        "addc.cc.u32 %5, %5, t5;\n\t"
        "addc.cc.u32 %6, %6, t6;\n\t"
        "addc.cc.u32 %7, %7, t7;\n\t"
        "addc.cc.u32 %8, %8, t8;\n\t"
        "addc.cc.u32 %9, %9, t9;\n\t"
        "addc.cc.u32 %10, %10, t10;\n\t"
        "addc.cc.u32 %11, %11, t11;\n\t"
        "addc.u32 %12, %12, 0;\n\t"
        "mul.wide.u16 t4, %18, %38;\n\t"
        "mul.wide.u16 t5, %20, %38;\n\t"
        "mul.wide.u16 t6, %22, %38;\n\t"
        "mul.wide.u16 t7, %24, %38;\n\t"
        "mul.wide.u16 t8, %23, %41;\n\t"
        "mul.wide.u16 t9, %25, %41;\n\t"
        "mul.wide.u16 t10, %27, %41;\n\t"
        "mul.wide.u16 t11, %29, %41;\n\t"
        "add.cc.u32 %4, %4, t4;\n\t"
        "addc.cc.u32 %5, %5, t5;\n\t"
        "addc.cc.u32 %6, %6, t6;\n\t"
        "addc.cc.u32 %7, %7, t7;\n\t"
        "addc.cc.u32 %8, %8, t8;\n\t"
        "addc.cc.u32 %9, %9, t9;\n\t"
        "addc.cc.u32 %10, %10, t10;\n\t"
        "addc.cc.u32 %11, %11, t11;\n\t"
        "addc.u32 %12, %12, 0;\n\t"
        "mul.wide.u16 t3, %17, %37;\n\t"
        "mul.wide.u16 t4, %19, %37;\n\t"
        "mul.wide.u16 t5, %21, %37;\n\t"
        "mul.wide.u16 t6, %23, %37;\n\t"
        "mul.wide.u16 t7, %25, %37;\n\t"
        "mul.wide.u16 t8, %22, %42;\n\t"
        "mul.wide.u16 t9, %24, %42;\n\t"
        "mul.wide.u16 t10, %26, %42;\n\t"
        "mul.wide.u16 t11, %28, %42;\n\t"
        "mul.wide.u16 t12, %30, %42;\n\t"
        "add.cc.u32 %3, %3, t3;\n\t"
        "addc.cc.u32 %4, %4, t4;\n\t"
        "addc.cc.u32 %5, %5, t5;\n\t"
        "addc.cc.u32 %6, %6, t6;\n\t"
        "addc.cc.u32 %7, %7, t7;\n\t"
        "addc.cc.u32 %8, %8, t8;\n\t"
        "addc.cc.u32 %9, %9, t9;\n\t"
        "addc.cc.u32 %10, %10, t10;\n\t"
        "addc.cc.u32 %11, %11, t11;\n\t"
        "addc.cc.u32 %12, %12, t12;\n\t"
        "addc.u32 %13, %13, 0;\n\t"
        "mul.wide.u16 t3, %18, %36;\n\t"
        "mul.wide.u16 t4, %20, %36;\n\t"
        "mul.wide.u16 t5, %22, %36;\n\t"
        "mul.wide.u16 t6, %24, %36;\n\t"
        "mul.wide.u16 t7, %26, %36;\n\t"
        "mul.wide.u16 t8, %21, %43;\n\t"
        "mul.wide.u16 t9, %23, %43;\n\t"
        "mul.wide.u16 t10, %25, %43;\n\t"
        "mul.wide.u16 t11, %27, %43;\n\t"
        "mul.wide.u16 t12, %29, %43;\n\t"
        "add.cc.u32 %3, %3, t3;\n\t"
        "addc.cc.u32 %4, %4, t4;\n\t"
        "addc.cc.u32 %5, %5, t5;\n\t"
        "addc.cc.u32 %6, %6, t6;\n\t"
        "addc.cc.u32 %7, %7, t7;\n\t"
        "addc.cc.u32 %8, %8, t8;\n\t"
        "addc.cc.u32 %9, %9, t9;\n\t"
        "addc.cc.u32 %10, %10, t10;\n\t"
        "addc.cc.u32 %11, %11, t11;\n\t"
        "addc.cc.u32 %12, %12, t12;\n\t"
        "addc.u32 %13, %13, 0;\n\t"
        "mul.wide.u16 t2, %17, %35;\n\t"
        "mul.wide.u16 t3, %19, %35;\n\t"
        "mul.wide.u16 t4, %21, %35;\n\t"
        "mul.wide.u16 t5, %23, %35;\n\t"
        "mul.wide.u16 t6, %25, %35;\n\t"
        "mul.wide.u16 t7, %27, %35;\n\t"
        "mul.wide.u16 t8, %20, %44;\n\t"
        "mul.wide.u16 t9, %22, %44;\n\t"
        "mul.wide.u16 t10, %24, %44;\n\t"
        "mul.wide.u16 t11, %26, %44;\n\t"
        "mul.wide.u16 t12, %28, %44;\n\t"
        "mul.wide.u16 t13, %30, %44;\n\t"
        "add.cc.u32 %2, %2, t2;\n\t"
        "addc.cc.u32 %3, %3, t3;\n\t"
        "addc.cc.u32 %4, %4, t4;\n\t"
        "addc.cc.u32 %5, %5, t5;\n\t"
        "addc.cc.u32 %6, %6, t6;\n\t"
        "addc.cc.u32 %7, %7, t7;\n\t"
        "addc.cc.u32 %8, %8, t8;\n\t"
        "addc.cc.u32 %9, %9, t9;\n\t"
        "addc.cc.u32 %10, %10, t10;\n\t"
        "addc.cc.u32 %11, %11, t11;\n\t"
        "addc.cc.u32 %12, %12, t12;\n\t"
        "addc.cc.u32 %13, %13, t13;\n\t"
        "addc.u32 %14, %14, 0;\n\t"
        "mul.wide.u16 t2, %18, %34;\n\t"
        "mul.wide.u16 t3, %20, %34;\n\t"
        "mul.wide.u16 t4, %22, %34;\n\t"
        "mul.wide.u16 t5, %24, %34;\n\t"
        "mul.wide.u16 t6, %26, %34;\n\t"
        "mul.wide.u16 t7, %28, %34;\n\t"
        "mul.wide.u16 t8, %19, %45;\n\t"
        "mul.wide.u16 t9, %21, %45;\n\t"
        "mul.wide.u16 t10, %23, %45;\n\t"
        "mul.wide.u16 t11, %25, %45;\n\t"
        "mul.wide.u16 t12, %27, %45;\n\t"
        "mul.wide.u16 t13, %29, %45;\n\t"
        "add.cc.u32 %2, %2, t2;\n\t"
        "addc.cc.u32 %3, %3, t3;\n\t"
        "addc.cc.u32 %4, %4, t4;\n\t"
        "addc.cc.u32 %5, %5, t5;\n\t"
        "addc.cc.u32 %6, %6, t6;\n\t"
        "addc.cc.u32 %7, %7, t7;\n\t"
        "addc.cc.u32 %8, %8, t8;\n\t"
        "addc.cc.u32 %9, %9, t9;\n\t"
        "addc.cc.u32 %10, %10, t10;\n\t"
        "addc.cc.u32 %11, %11, t11;\n\t"
        "addc.cc.u32 %12, %12, t12;\n\t"
        "addc.cc.u32 %13, %13, t13;\n\t"
        "addc.u32 %14, %14, 0;\n\t"
        "mul.wide.u16 t1, %17, %33;\n\t"
        "mul.wide.u16 t2, %19, %33;\n\t"
        "mul.wide.u16 t3, %21, %33;\n\t"
        "mul.wide.u16 t4, %23, %33;\n\t"
        "mul.wide.u16 t5, %25, %33;\n\t"
        "mul.wide.u16 t6, %27, %33;\n\t"
        "mul.wide.u16 t7, %29, %33;\n\t"
        "mul.wide.u16 t8, %18, %46;\n\t"
        "mul.wide.u16 t9, %20, %46;\n\t"
        "mul.wide.u16 t10, %22, %46;\n\t"
        "mul.wide.u16 t11, %24, %46;\n\t"
        "mul.wide.u16 t12, %26, %46;\n\t"
        "mul.wide.u16 t13, %28, %46;\n\t"
        "mul.wide.u16 t14, %30, %46;\n\t"
        "add.cc.u32 %1, %1, t1;\n\t"
        "addc.cc.u32 %2, %2, t2;\n\t"
        "addc.cc.u32 %3, %3, t3;\n\t"
        "addc.cc.u32 %4, %4, t4;\n\t"
        "addc.cc.u32 %5, %5, t5;\n\t"
        "addc.cc.u32 %6, %6, t6;\n\t"
        "addc.cc.u32 %7, %7, t7;\n\t"
        "addc.cc.u32 %8, %8, t8;\n\t"
        "addc.cc.u32 %9, %9, t9;\n\t"
        "addc.cc.u32 %10, %10, t10;\n\t"
        "addc.cc.u32 %11, %11, t11;\n\t"
        "addc.cc.u32 %12, %12, t12;\n\t"
        "addc.cc.u32 %13, %13, t13;\n\t"
        "addc.cc.u32 %14, %14, t14;\n\t"
        "addc.u32 %15, %15, 0;\n\t"
        "mul.wide.u16 t1, %18, %32;\n\t"
        "mul.wide.u16 t2, %20, %32;\n\t"
        "mul.wide.u16 t3, %22, %32;\n\t"
        "mul.wide.u16 t4, %24, %32;\n\t"
        "mul.wide.u16 t5, %26, %32;\n\t"
        "mul.wide.u16 t6, %28, %32;\n\t"
        "mul.wide.u16 t7, %30, %32;\n\t"
        "mul.wide.u16 t8, %17, %47;\n\t"
        "mul.wide.u16 t9, %19, %47;\n\t"
        "mul.wide.u16 t10, %21, %47;\n\t"
        "mul.wide.u16 t11, %23, %47;\n\t"
        "mul.wide.u16 t12, %25, %47;\n\t"
        "mul.wide.u16 t13, %27, %47;\n\t"
        "mul.wide.u16 t14, %29, %47;\n\t"
        "add.cc.u32 %1, %1, t1;\n\t"
        "addc.cc.u32 %2, %2, t2;\n\t"
        "addc.cc.u32 %3, %3, t3;\n\t"
        "addc.cc.u32 %4, %4, t4;\n\t"
        "addc.cc.u32 %5, %5, t5;\n\t"
        "addc.cc.u32 %6, %6, t6;\n\t"
        "addc.cc.u32 %7, %7, t7;\n\t"
        "addc.cc.u32 %8, %8, t8;\n\t"
        "addc.cc.u32 %9, %9, t9;\n\t"
        "addc.cc.u32 %10, %10, t10;\n\t"
        "addc.cc.u32 %11, %11, t11;\n\t"
        "addc.cc.u32 %12, %12, t12;\n\t"
        "addc.cc.u32 %13, %13, t13;\n\t"
        "addc.cc.u32 %14, %14, t14;\n\t"
        "addc.u32 %15, %15, 0;\n\t"
        // Compute second partial sum
        "mul.wide.u16 t0, %16, %33;\n\t"
        "mul.wide.u16 t1, %16, %35;\n\t"
        "mul.wide.u16 t2, %16, %37;\n\t"
        "mul.wide.u16 t3, %16, %39;\n\t"
        "mul.wide.u16 t4, %16, %41;\n\t"
        "mul.wide.u16 t5, %16, %43;\n\t"
        "mul.wide.u16 t6, %16, %45;\n\t"
        "mul.wide.u16 t7, %16, %47;\n\t"
        "mul.wide.u16 t8, %31, %34;\n\t"
        "mul.wide.u16 t9, %31, %36;\n\t"
        "mul.wide.u16 t10, %31, %38;\n\t"
        "mul.wide.u16 t11, %31, %40;\n\t"
        "mul.wide.u16 t12, %31, %42;\n\t"
        "mul.wide.u16 t13, %31, %44;\n\t"
        "mul.wide.u16 t14, %31, %46;\n\t"
        "mul.wide.u16 s7, %17, %46;\n\t"
        "add.cc.u32 t7, t7, s7;\n\t"
        "addc.u32 t8, t8, 0;\n\t"
        "mul.wide.u16 s7, %18, %45;\n\t"
        "add.cc.u32 t7, t7, s7;\n\t"
        "addc.u32 t8, t8, 0;\n\t"
        "mul.wide.u16 s6, %17, %44;\n\t"
        "mul.wide.u16 s7, %19, %44;\n\t"
        "mul.wide.u16 s8, %30, %35;\n\t"
        "add.cc.u32 t6, t6, s6;\n\t"
        "addc.cc.u32 t7, t7, s7;\n\t"
        "addc.cc.u32 t8, t8, s8;\n\t"
        "addc.u32 t9, t9, 0;\n\t"
        "mul.wide.u16 s6, %18, %43;\n\t"
        "mul.wide.u16 s7, %20, %43;\n\t"
        "mul.wide.u16 s8, %29, %36;\n\t"
        "add.cc.u32 t6, t6, s6;\n\t"
        "addc.cc.u32 t7, t7, s7;\n\t"
        "addc.cc.u32 t8, t8, s8;\n\t"
        "addc.u32 t9, t9, 0;\n\t"
        "mul.wide.u16 s5, %17, %42;\n\t"
        "mul.wide.u16 s6, %19, %42;\n\t"
        "mul.wide.u16 s7, %21, %42;\n\t"
        "mul.wide.u16 s8, %28, %37;\n\t"
        "mul.wide.u16 s9, %30, %37;\n\t"
        "add.cc.u32 t5, t5, s5;\n\t"
        "addc.cc.u32 t6, t6, s6;\n\t"
        "addc.cc.u32 t7, t7, s7;\n\t"
        "addc.cc.u32 t8, t8, s8;\n\t"
        "addc.cc.u32 t9, t9, s9;\n\t"
        "addc.u32 t10, t10, 0;\n\t"
        "mul.wide.u16 s5, %18, %41;\n\t"
        "mul.wide.u16 s6, %20, %41;\n\t"
        "mul.wide.u16 s7, %22, %41;\n\t"
        "mul.wide.u16 s8, %27, %38;\n\t"
        "mul.wide.u16 s9, %29, %38;\n\t"
        "add.cc.u32 t5, t5, s5;\n\t"
        "addc.cc.u32 t6, t6, s6;\n\t"
        "addc.cc.u32 t7, t7, s7;\n\t"
        "addc.cc.u32 t8, t8, s8;\n\t"
        "addc.cc.u32 t9, t9, s9;\n\t"
        "addc.u32 t10, t10, 0;\n\t"
        "mul.wide.u16 s4, %17, %40;\n\t"
        "mul.wide.u16 s5, %19, %40;\n\t"
        "mul.wide.u16 s6, %21, %40;\n\t"
        "mul.wide.u16 s7, %23, %40;\n\t"
        "mul.wide.u16 s8, %26, %39;\n\t"
        "mul.wide.u16 s9, %28, %39;\n\t"
        "mul.wide.u16 s10, %30, %39;\n\t"
        "add.cc.u32 t4, t4, s4;\n\t"
        "addc.cc.u32 t5, t5, s5;\n\t"
        "addc.cc.u32 t6, t6, s6;\n\t"
        "addc.cc.u32 t7, t7, s7;\n\t"
        "addc.cc.u32 t8, t8, s8;\n\t"
        "addc.cc.u32 t9, t9, s9;\n\t"
        "addc.cc.u32 t10, t10, s10;\n\t"
        "addc.u32 t11, t11, 0;\n\t"
        "mul.wide.u16 s4, %18, %39;\n\t"
        "mul.wide.u16 s5, %20, %39;\n\t"
        "mul.wide.u16 s6, %22, %39;\n\t"
        "mul.wide.u16 s7, %24, %39;\n\t"
        "mul.wide.u16 s8, %25, %40;\n\t"
        "mul.wide.u16 s9, %27, %40;\n\t"
        "mul.wide.u16 s10, %29, %40;\n\t"
        "add.cc.u32 t4, t4, s4;\n\t"
        "addc.cc.u32 t5, t5, s5;\n\t"
        "addc.cc.u32 t6, t6, s6;\n\t"
        "addc.cc.u32 t7, t7, s7;\n\t"
        "addc.cc.u32 t8, t8, s8;\n\t"
        "addc.cc.u32 t9, t9, s9;\n\t"
        "addc.cc.u32 t10, t10, s10;\n\t"
        "addc.u32 t11, t11, 0;\n\t"
        "mul.wide.u16 s3, %17, %38;\n\t"
        "mul.wide.u16 s4, %19, %38;\n\t"
        "mul.wide.u16 s5, %21, %38;\n\t"
        "mul.wide.u16 s6, %23, %38;\n\t"
        "mul.wide.u16 s7, %25, %38;\n\t"
        "mul.wide.u16 s8, %24, %41;\n\t"
        "mul.wide.u16 s9, %26, %41;\n\t"
        "mul.wide.u16 s10, %28, %41;\n\t"
        "mul.wide.u16 s11, %30, %41;\n\t"
        "add.cc.u32 t3, t3, s3;\n\t"
        "addc.cc.u32 t4, t4, s4;\n\t"
        "addc.cc.u32 t5, t5, s5;\n\t"
        "addc.cc.u32 t6, t6, s6;\n\t"
        "addc.cc.u32 t7, t7, s7;\n\t"
        "addc.cc.u32 t8, t8, s8;\n\t"
        "addc.cc.u32 t9, t9, s9;\n\t"
        "addc.cc.u32 t10, t10, s10;\n\t"
        "addc.cc.u32 t11, t11, s11;\n\t"
        "addc.u32 t12, t12, 0;\n\t"
        "mul.wide.u16 s3, %18, %37;\n\t"
        "mul.wide.u16 s4, %20, %37;\n\t"
        "mul.wide.u16 s5, %22, %37;\n\t"
        "mul.wide.u16 s6, %24, %37;\n\t"
        "mul.wide.u16 s7, %26, %37;\n\t"
        "mul.wide.u16 s8, %23, %42;\n\t"
        "mul.wide.u16 s9, %25, %42;\n\t"
        "mul.wide.u16 s10, %27, %42;\n\t"
        "mul.wide.u16 s11, %29, %42;\n\t"
        "add.cc.u32 t3, t3, s3;\n\t"
        "addc.cc.u32 t4, t4, s4;\n\t"
        "addc.cc.u32 t5, t5, s5;\n\t"
        "addc.cc.u32 t6, t6, s6;\n\t"
        "addc.cc.u32 t7, t7, s7;\n\t"
        "addc.cc.u32 t8, t8, s8;\n\t"
        "addc.cc.u32 t9, t9, s9;\n\t"
        "addc.cc.u32 t10, t10, s10;\n\t"
        "addc.cc.u32 t11, t11, s11;\n\t"
        "addc.u32 t12, t12, 0;\n\t"
        "mul.wide.u16 s2, %17, %36;\n\t"
        "mul.wide.u16 s3, %19, %36;\n\t"
        "mul.wide.u16 s4, %21, %36;\n\t"
        "mul.wide.u16 s5, %23, %36;\n\t"
        "mul.wide.u16 s6, %25, %36;\n\t"
        "mul.wide.u16 s7, %27, %36;\n\t"
        "mul.wide.u16 s8, %22, %43;\n\t"
        "mul.wide.u16 s9, %24, %43;\n\t"
        "mul.wide.u16 s10, %26, %43;\n\t"
        "mul.wide.u16 s11, %28, %43;\n\t"
        "mul.wide.u16 s12, %30, %43;\n\t"
        "add.cc.u32 t2, t2, s2;\n\t"
        "addc.cc.u32 t3, t3, s3;\n\t"
        "addc.cc.u32 t4, t4, s4;\n\t"
        "addc.cc.u32 t5, t5, s5;\n\t"
        "addc.cc.u32 t6, t6, s6;\n\t"
        "addc.cc.u32 t7, t7, s7;\n\t"
        "addc.cc.u32 t8, t8, s8;\n\t"
        "addc.cc.u32 t9, t9, s9;\n\t"
        "addc.cc.u32 t10, t10, s10;\n\t"
        "addc.cc.u32 t11, t11, s11;\n\t"
        "addc.cc.u32 t12, t12, s12;\n\t"
        "addc.u32 t13, t13, 0;\n\t"
        "mul.wide.u16 s2, %18, %35;\n\t"
        "mul.wide.u16 s3, %20, %35;\n\t"
        "mul.wide.u16 s4, %22, %35;\n\t"
        "mul.wide.u16 s5, %24, %35;\n\t"
        "mul.wide.u16 s6, %26, %35;\n\t"
        "mul.wide.u16 s7, %28, %35;\n\t"
        "mul.wide.u16 s8, %21, %44;\n\t"
        "mul.wide.u16 s9, %23, %44;\n\t"
        "mul.wide.u16 s10, %25, %44;\n\t"
        "mul.wide.u16 s11, %27, %44;\n\t"
        "mul.wide.u16 s12, %29, %44;\n\t"
        "add.cc.u32 t2, t2, s2;\n\t"
        "addc.cc.u32 t3, t3, s3;\n\t"
        "addc.cc.u32 t4, t4, s4;\n\t"
        "addc.cc.u32 t5, t5, s5;\n\t"
        "addc.cc.u32 t6, t6, s6;\n\t"
        "addc.cc.u32 t7, t7, s7;\n\t"
        "addc.cc.u32 t8, t8, s8;\n\t"
        "addc.cc.u32 t9, t9, s9;\n\t"
        "addc.cc.u32 t10, t10, s10;\n\t"
        "addc.cc.u32 t11, t11, s11;\n\t"
        "addc.cc.u32 t12, t12, s12;\n\t"
        "addc.u32 t13, t13, 0;\n\t"
        "mul.wide.u16 s1, %17, %34;\n\t"
        "mul.wide.u16 s2, %19, %34;\n\t"
        "mul.wide.u16 s3, %21, %34;\n\t"
        "mul.wide.u16 s4, %23, %34;\n\t"
        "mul.wide.u16 s5, %25, %34;\n\t"
        "mul.wide.u16 s6, %27, %34;\n\t"
        "mul.wide.u16 s7, %29, %34;\n\t"
        "mul.wide.u16 s8, %20, %45;\n\t"
        "mul.wide.u16 s9, %22, %45;\n\t"
        "mul.wide.u16 s10, %24, %45;\n\t"
        "mul.wide.u16 s11, %26, %45;\n\t"
        "mul.wide.u16 s12, %28, %45;\n\t"
        "mul.wide.u16 s13, %30, %45;\n\t"
        "add.cc.u32 t1, t1, s1;\n\t"
        "addc.cc.u32 t2, t2, s2;\n\t"
        "addc.cc.u32 t3, t3, s3;\n\t"
        "addc.cc.u32 t4, t4, s4;\n\t"
        "addc.cc.u32 t5, t5, s5;\n\t"
        "addc.cc.u32 t6, t6, s6;\n\t"
        "addc.cc.u32 t7, t7, s7;\n\t"
        "addc.cc.u32 t8, t8, s8;\n\t"
        "addc.cc.u32 t9, t9, s9;\n\t"
        "addc.cc.u32 t10, t10, s10;\n\t"
        "addc.cc.u32 t11, t11, s11;\n\t"
        "addc.cc.u32 t12, t12, s12;\n\t"
        "addc.cc.u32 t13, t13, s13;\n\t"
        "addc.u32 t14, t14, 0;\n\t"
        "mul.wide.u16 s1, %18, %33;\n\t"
        "mul.wide.u16 s2, %20, %33;\n\t"
        "mul.wide.u16 s3, %22, %33;\n\t"
        "mul.wide.u16 s4, %24, %33;\n\t"
        "mul.wide.u16 s5, %26, %33;\n\t"
        "mul.wide.u16 s6, %28, %33;\n\t"
        "mul.wide.u16 s7, %30, %33;\n\t"
        "mul.wide.u16 s8, %19, %46;\n\t"
        "mul.wide.u16 s9, %21, %46;\n\t"
        "mul.wide.u16 s10, %23, %46;\n\t"
        "mul.wide.u16 s11, %25, %46;\n\t"
        "mul.wide.u16 s12, %27, %46;\n\t"
        "mul.wide.u16 s13, %29, %46;\n\t"
        "add.cc.u32 t1, t1, s1;\n\t"
        "addc.cc.u32 t2, t2, s2;\n\t"
        "addc.cc.u32 t3, t3, s3;\n\t"
        "addc.cc.u32 t4, t4, s4;\n\t"
        "addc.cc.u32 t5, t5, s5;\n\t"
        "addc.cc.u32 t6, t6, s6;\n\t"
        "addc.cc.u32 t7, t7, s7;\n\t"
        "addc.cc.u32 t8, t8, s8;\n\t"
        "addc.cc.u32 t9, t9, s9;\n\t"
        "addc.cc.u32 t10, t10, s10;\n\t"
        "addc.cc.u32 t11, t11, s11;\n\t"
        "addc.cc.u32 t12, t12, s12;\n\t"
        "addc.cc.u32 t13, t13, s13;\n\t"
        "addc.u32 t14, t14, 0;\n\t"
        "mul.wide.u16 s0, %17, %32;\n\t"
        "mul.wide.u16 s1, %19, %32;\n\t"
        "mul.wide.u16 s2, %21, %32;\n\t"
        "mul.wide.u16 s3, %23, %32;\n\t"
        "mul.wide.u16 s4, %25, %32;\n\t"
        "mul.wide.u16 s5, %27, %32;\n\t"
        "mul.wide.u16 s6, %29, %32;\n\t"
        "mul.wide.u16 s7, %31, %32;\n\t"
        "mul.wide.u16 s8, %18, %47;\n\t"
        "mul.wide.u16 s9, %20, %47;\n\t"
        "mul.wide.u16 s10, %22, %47;\n\t"
        "mul.wide.u16 s11, %24, %47;\n\t"
        "mul.wide.u16 s12, %26, %47;\n\t"
        "mul.wide.u16 s13, %28, %47;\n\t"
        "mul.wide.u16 s14, %30, %47;\n\t"
        "add.cc.u32 t0, t0, s0;\n\t"
        "addc.cc.u32 t1, t1, s1;\n\t"
        "addc.cc.u32 t2, t2, s2;\n\t"
        "addc.cc.u32 t3, t3, s3;\n\t"
        "addc.cc.u32 t4, t4, s4;\n\t"
        "addc.cc.u32 t5, t5, s5;\n\t"
        "addc.cc.u32 t6, t6, s6;\n\t"
        "addc.cc.u32 t7, t7, s7;\n\t"
        "addc.cc.u32 t8, t8, s8;\n\t"
        "addc.cc.u32 t9, t9, s9;\n\t"
        "addc.cc.u32 t10, t10, s10;\n\t"
        "addc.cc.u32 t11, t11, s11;\n\t"
        "addc.cc.u32 t12, t12, s12;\n\t"
        "addc.cc.u32 t13, t13, s13;\n\t"
        "addc.cc.u32 t14, t14, s14;\n\t"
        "addc.u32 t15, 0, 0;\n\t"
        // offset second partial sum by 16 bits
        "shf.l.clamp.b32 s15, t14, t15, 16;\n\t"
        "shf.l.clamp.b32 s14, t13, t14, 16;\n\t"
        "shf.l.clamp.b32 s13, t12, t13, 16;\n\t"
        "shf.l.clamp.b32 s12, t11, t12, 16;\n\t"
        "shf.l.clamp.b32 s11, t10, t11, 16;\n\t"
        "shf.l.clamp.b32 s10, t9, t10, 16;\n\t"
        "shf.l.clamp.b32 s9, t8, t9, 16;\n\t"
        "shf.l.clamp.b32 s8, t7, t8, 16;\n\t"
        "shf.l.clamp.b32 s7, t6, t7, 16;\n\t"
        "shf.l.clamp.b32 s6, t5, t6, 16;\n\t"
        "shf.l.clamp.b32 s5, t4, t5, 16;\n\t"
        "shf.l.clamp.b32 s4, t3, t4, 16;\n\t"
        "shf.l.clamp.b32 s3, t2, t3, 16;\n\t"
        "shf.l.clamp.b32 s2, t1, t2, 16;\n\t"
        "shf.l.clamp.b32 s1, t0, t1, 16;\n\t"
        "shf.l.clamp.b32 s0, 0, t0, 16;\n\t"
        // Add partial sums
        "add.cc.u32 %0, %0, s0;\n\t"
        "addc.cc.u32 %1, %1, s1;\n\t"
        "addc.cc.u32 %2, %2, s2;\n\t"
        "addc.cc.u32 %3, %3, s3;\n\t"
        "addc.cc.u32 %4, %4, s4;\n\t"
        "addc.cc.u32 %5, %5, s5;\n\t"
        "addc.cc.u32 %6, %6, s6;\n\t"
        "addc.cc.u32 %7, %7, s7;\n\t"
        "addc.cc.u32 %8, %8, s8;\n\t"
        "addc.cc.u32 %9, %9, s9;\n\t"
        "addc.cc.u32 %10, %10, s10;\n\t"
        "addc.cc.u32 %11, %11, s11;\n\t"
        "addc.cc.u32 %12, %12, s12;\n\t"
        "addc.cc.u32 %13, %13, s13;\n\t"
        "addc.cc.u32 %14, %14, s14;\n\t"
        "addc.u32 %15, %15, s15;\n\t"
     "}"
     : "=r"(res.u32[0]), "=r"(res.u32[1]), "=r"(res.u32[2]), "=r"(res.u32[3]), "=r"(res.u32[4]), "=r"(res.u32[5]), "=r"(res.u32[6]), "=r"(res.u32[7]), "=r"(res.u32[8]), "=r"(res.u32[9]), "=r"(res.u32[10]), "=r"(res.u32[11]), "=r"(res.u32[12]), "=r"(res.u32[13]), "=r"(res.u32[14]), "=r"(res.u32[15])
     : "h"(a.u16[0]), "h"(a.u16[1]), "h"(a.u16[2]), "h"(a.u16[3]), "h"(a.u16[4]), "h"(a.u16[5]), "h"(a.u16[6]), "h"(a.u16[7]), "h"(a.u16[8]), "h"(a.u16[9]), "h"(a.u16[10]), "h"(a.u16[11]), "h"(a.u16[12]), "h"(a.u16[13]), "h"(a.u16[14]), "h"(a.u16[15]),"h"(b.u16[0]), "h"(b.u16[1]), "h"(b.u16[2]), "h"(b.u16[3]),"h"(b.u16[4]), "h"(b.u16[5]), "h"(b.u16[6]), "h"(b.u16[7]),"h"(b.u16[8]), "h"(b.u16[9]), "h"(b.u16[10]), "h"(b.u16[11]),"h"(b.u16[12]), "h"(b.u16[13]), "h"(b.u16[14]), "h"(b.u16[15]));
     
    y[0] = res.u32[0];
    y[1] = res.u32[1];
    y[2] = res.u32[2];
    y[3] = res.u32[3];
    y[4] = res.u32[4];
    y[5] = res.u32[5];
    y[6] = res.u32[6];
    y[7] = res.u32[7];

    x[0] = res.u32[8];
    x[1] = res.u32[9];
    x[2] = res.u32[10];
    x[3] = res.u32[11];
    x[4] = res.u32[12];
    x[5] = res.u32[13];
    x[6] = res.u32[14];
    x[7] = res.u32[15];
}

/* Computes Z = X*Y mod P. */
/* Output fits into 8 words but could be greater than P */
__device__ __forceinline__ void mul_reduce(uint32 *Z, const uint32 *X, const uint32 *Y)
{
    uint32 a[8], b[8];

    copy(a, X);
    copy(b, Y);

    umul256wide(a, b);

    //printf("a: {%d %d %d %d %d %d %d %d}\n", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);
    //printf("b: {%d %d %d %d %d %d %d %d}\n", b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);

    asm("{                                        \n\t"
        "mad.lo.cc.u32  %0, 38, %8, %16;          \n\t"
        "madc.hi.u32    %16, 38, %8, 0;           \n\t"
        "mad.lo.cc.u32  %1, 38, %9, %16;          \n\t"
        "madc.hi.u32    %16, 38, %9, 0;           \n\t"
        "add.cc.u32     %1, %1, %17;              \n\t"
        "madc.lo.cc.u32  %2, 38, %10, %16;         \n\t"
        "madc.hi.u32    %16, 38, %10, 0;          \n\t"
        "add.cc.u32     %2, %2, %18;              \n\t"
        "madc.lo.cc.u32  %3, 38, %11, %16;         \n\t"
        "madc.hi.u32    %16, 38, %11, 0;          \n\t"
        "add.cc.u32     %3, %3, %19;              \n\t"
        "madc.lo.cc.u32  %4, 38, %12, %16;         \n\t"
        "madc.hi.u32    %16, 38, %12, 0;          \n\t"
        "add.cc.u32     %4, %4, %20;              \n\t"
        "madc.lo.cc.u32  %5, 38, %13, %16;         \n\t"
        "madc.hi.u32    %16, 38, %13, 0;          \n\t"
        "add.cc.u32     %5, %5, %21;              \n\t"
        "madc.lo.cc.u32  %6, 38, %14, %16;         \n\t"
        "madc.hi.u32    %16, 38, %14, 0;          \n\t"
        "add.cc.u32     %6, %6, %22;              \n\t"
        "madc.lo.cc.u32  %7, 38, %15, %16;         \n\t"
        "madc.hi.u32    %16, 38, %15, 0;          \n\t"
        "add.cc.u32     %7, %7, %23;              \n\t"
        "addc.u32       %16, %16, 0;              \n\t"
        "mul.lo.u32     %16, %16, 38;             \n\t"
        "add.cc.u32     %0, %0, %16;              \n\t"
        "addc.u32       %1, %1,  0;               \n\t"
        "}"
        // %0 - %7
        : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3]), "=r"(Z[4]), "=r"(Z[5]), "=r"(Z[6]), "=r"(Z[7])
        // %8 - %15
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
        // %16 - %23
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7])
        );
}


// Assembly directives
#define UADDO(c, a, b) asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADDC(c, a, b) asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADD(c, a, b) asm volatile ("addc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define UADDO1(c, a) asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADDC1(c, a) asm volatile ("addc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADD1(c, a) asm volatile ("addc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));

#define USUBO(c, a, b) asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUBC(c, a, b) asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUB(c, a, b) asm volatile ("subc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define USUBO1(c, a) asm volatile ("sub.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUBC1(c, a) asm volatile ("subc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUB1(c, a) asm volatile ("subc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) );

#define UMULLO(lo,a, b) asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
#define UMULHI(hi,a, b) asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
#define MADDO(r,a,b,c) asm volatile ("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory" );
#define MADDC(r,a,b,c) asm volatile ("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory" );
#define MADD(r,a,b,c) asm volatile ("madc.hi.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));


#define UMult(r, a, b) {\
  UMULLO(r[0],a[4],b); \
  UMULLO(r[1],a[5],b); \
  MADDO(r[1], a[4],b,r[1]); \
  UMULLO(r[2],a[6], b); \
  MADDC(r[2], a[5], b, r[2]); \
  UMULLO(r[3],a[7], b); \
  MADDC(r[3], a[6], b, r[3]); \
  MADD(r[4], a[7], b, 0ULL);}

/* Computes Z = X*Y mod P. */
/* Output fits into 8 words but could be greater than P */
__device__ __forceinline__ void mul_reduce_4(uint32 *Z, const uint32 *X, const uint32 *Y)
{
    uint32 a[8], b[8];

    t512 r512;

    t256 r;

    uint64 t[5];
    uint64 ah, al;

    copy(a, X);
    copy(b, Y);

    umul256wide(a, b);

    r512.u32[0] = b[0];
    r512.u32[1] = b[1];
    r512.u32[2] = b[2];
    r512.u32[3] = b[3];
    r512.u32[4] = b[4];
    r512.u32[5] = b[5];
    r512.u32[6] = b[6];
    r512.u32[7] = b[7];

    r512.u32[8] = a[0];
    r512.u32[9] = a[1];
    r512.u32[10] = a[2];
    r512.u32[11] = a[3];
    r512.u32[12] = a[4];
    r512.u32[13] = a[5];
    r512.u32[14] = a[6];
    r512.u32[15] = a[7];

    // Reduce from 512 to 320 
    UMult(t, r512.u64, (uint64)38);
    UADDO1(r512.u64[0],t[0]);
    UADDC1(r512.u64[1],t[1]);
    UADDC1(r512.u64[2],t[2]);
    UADDC1(r512.u64[3],t[3]);

    // Reduce from 320 to 256
    UADD1(t[4],0ULL);
    UMULLO(al,t[4],(uint64)38);
    UMULHI(ah,t[4],(uint64)38);
    UADDO(r.u64[0],r512.u64[0],al);
    UADDC(r.u64[1],r512.u64[1],ah);
    UADDC(r.u64[2],r512.u64[2],0ULL);
    UADD(r.u64[3],r512.u64[3],0ULL);

    copy(Z, r.u32);
}

__device__ __forceinline__ void mul_reduce_3(uint32 *Z, const uint32 *X, const uint32 *Y)
{
    uint32 a[8], b[8];

    copy(a, X);
    copy(b, Y);

    int i;
    int j;
    uint64 c;
    uint64 temp;
    uint32 u[16];
  
    //Initialize variables
    temp = 0;
    c = 0;
  
    //Comba's method is used to perform multiplication
    for(i = 0; i < 16; i++)
    {
       //The algorithm computes the products, column by column
       if(i < 8)
       {
          //Inner loop
          for(j = 0; j <= i; j++)
          {
             temp += (uint64) a[j] * b[i - j];
             c += temp >> 32;
             temp &= 0xFFFFFFFF;
          }
       }
       else
       {
          //Inner loop
          for(j = i - 7; j < 8; j++)
          {
             temp += (uint64) a[j] * b[i - j];
             c += temp >> 32;
             temp &= 0xFFFFFFFF;
          }
       }
  
       //At the bottom of each column, the final result is written to memory
       u[i] = temp & 0xFFFFFFFF;
  
       //Propagate the carry upwards
       temp = c & 0xFFFFFFFF;
       c >>= 32;
    }

    asm("{                                        \n\t"
        "mad.lo.cc.u32  %0, 38, %8, %16;          \n\t"
        "madc.hi.u32    %16, 38, %8, 0;           \n\t"
        "mad.lo.cc.u32  %1, 38, %9, %16;          \n\t"
        "madc.hi.u32    %16, 38, %9, 0;           \n\t"
        "add.cc.u32     %1, %1, %17;              \n\t"
        "madc.lo.cc.u32  %2, 38, %10, %16;         \n\t"
        "madc.hi.u32    %16, 38, %10, 0;          \n\t"
        "add.cc.u32     %2, %2, %18;              \n\t"
        "madc.lo.cc.u32  %3, 38, %11, %16;         \n\t"
        "madc.hi.u32    %16, 38, %11, 0;          \n\t"
        "add.cc.u32     %3, %3, %19;              \n\t"
        "madc.lo.cc.u32  %4, 38, %12, %16;         \n\t"
        "madc.hi.u32    %16, 38, %12, 0;          \n\t"
        "add.cc.u32     %4, %4, %20;              \n\t"
        "madc.lo.cc.u32  %5, 38, %13, %16;         \n\t"
        "madc.hi.u32    %16, 38, %13, 0;          \n\t"
        "add.cc.u32     %5, %5, %21;              \n\t"
        "madc.lo.cc.u32  %6, 38, %14, %16;         \n\t"
        "madc.hi.u32    %16, 38, %14, 0;          \n\t"
        "add.cc.u32     %6, %6, %22;              \n\t"
        "madc.lo.cc.u32  %7, 38, %15, %16;         \n\t"
        "madc.hi.u32    %16, 38, %15, 0;          \n\t"
        "add.cc.u32     %7, %7, %23;              \n\t"
        "addc.u32       %16, %16, 0;              \n\t"
        "mul.lo.u32     %16, %16, 38;             \n\t"
        "add.cc.u32     %0, %0, %16;              \n\t"
        "addc.u32       %1, %1,  0;               \n\t"
        "}"
        // %0 - %7
        : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3]), "=r"(Z[4]), "=r"(Z[5]), "=r"(Z[6]), "=r"(Z[7])
        // %8 - %15
        : "r"(u[8]), "r"(u[9]), "r"(u[10]), "r"(u[11]), "r"(u[12]), "r"(u[13]), "r"(u[14]), "r"(u[15]),
        // %16 - %23
          "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]), "r"(u[4]), "r"(u[5]), "r"(u[6]), "r"(u[7])
        );
}


/* Computes Z = X*Y mod P. */
__device__ __forceinline__ void mul_mod(uint32* Z, const uint32* X, const uint32* Y)
{
    mul_reduce(Z, X, Y);
    mod_p(Z);
}

// multiply two unsigned 128-bit integers into an unsigned 256-bit product
__device__ __forceinline__ void umul128wide2(uint32 *c, const uint32 *a, const uint32 *b)
{
    asm("{\n\t"
         "mul.lo.u32      %0, %8, %12;\n\t"
         "mul.hi.u32      %1, %8, %12;\n\t"
         "mad.lo.cc.u32   %1, %8, %13, %1;\n\t"
         "madc.hi.u32     %2, %8, %13, 0;\n\t"
         "mad.lo.cc.u32   %1, %9, %12, %1;\n\t"
         "madc.hi.cc.u32  %2, %9, %12, %2;\n\t"
         "madc.hi.u32     %3, %8, %14, 0;\n\t"
         "mad.lo.cc.u32   %2, %8, %14, %2;\n\t"
         "madc.hi.cc.u32  %3, %9, %13, %3;\n\t"
         "madc.hi.u32     %4, %8, %15, 0;\n\t"
         "mad.lo.cc.u32   %2, %9, %13, %2;\n\t"
         "madc.hi.cc.u32  %3, %10, %12, %3;\n\t"
         "madc.hi.cc.u32  %4, %9, %14, %4;\n\t"
         "madc.hi.u32     %5, %9, %15, 0;\n\t"
         "mad.lo.cc.u32   %2, %10, %12, %2;\n\t"
         "madc.lo.cc.u32  %3, %8, %15, %3;\n\t"
         "madc.hi.cc.u32  %4, %10, %13, %4;\n\t"
         "madc.hi.cc.u32  %5, %10, %14, %5;\n\t"
         "madc.hi.u32     %6, %10, %15, 0;\n\t"
         "mad.lo.cc.u32   %3, %9, %14, %3;\n\t"
         "madc.hi.cc.u32  %4, %11, %12, %4;\n\t"
         "madc.hi.cc.u32  %5, %11, %13, %5;\n\t"
         "madc.hi.cc.u32  %6, %11, %14, %6;\n\t"
         "madc.hi.u32     %7, %11, %15, 0;\n\t"
         "mad.lo.cc.u32   %3, %10, %13, %3;\n\t"
         "madc.lo.cc.u32  %4, %9, %15, %4;\n\t"
         "madc.lo.cc.u32  %5, %10, %15, %5;\n\t"
         "madc.lo.cc.u32  %6, %11, %15, %6;\n\t"
         "addc.u32        %7, %7, 0;\n\t"
         "mad.lo.cc.u32   %3, %11, %12, %3;\n\t"
         "madc.lo.cc.u32  %4, %10, %14, %4;\n\t"
         "madc.lo.cc.u32  %5, %11, %14, %5;\n\t"
         "addc.cc.u32     %6, %6, 0;\n\t"
         "addc.u32        %7, %7, 0;\n\t"
         "mad.lo.cc.u32   %4, %11, %13, %4;\n\t"
         "addc.cc.u32     %5, %5, 0;\n\t"
         "addc.cc.u32     %6, %6, 0;\n\t"
         "addc.u32        %7, %7, 0;\n\t"
         "}"
         : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3]), "=r"(c[4]), "=r"(c[5]), "=r"(c[6]), "=r"(c[7])
         : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]));
}

__device__ __forceinline__ void umul128wide(uint32 *c, const uint16 *U, const uint16 *V){
    asm ("{\n\t"
         ".reg .u32 r0, r1, r2, r3, r4, r5, r6, r7;\n\t"
         ".reg .u32 t0, t1, t2, t3, t4, t5, t6, t7;\n\t"
         // compute first partial sum
         "mul.wide.u16    r0, %8, %16;\n\t"
         "mul.wide.u16    r1, %8, %18;\n\t"
         "mul.wide.u16    r2, %8, %20;\n\t"
         "mul.wide.u16    r3, %8, %22;\n\t"
         "mul.wide.u16    r4, %9, %23;\n\t"
         "mul.wide.u16    r5, %11, %23;\n\t"
         "mul.wide.u16    r6, %13, %23;\n\t"
         "mul.wide.u16    r7, %15, %23;\n\t"
         "mul.wide.u16    t3, %9, %21;\n\t"
         "mul.wide.u16    t4, %10, %22;\n\t"
         "add.cc.u32      r3, r3, t3;\n\t"
         "addc.cc.u32     r4, r4, t4;\n\t"
         "addc.u32        r5, r5, 0;\n\t"
         "mul.wide.u16    t3, %10, %20;\n\t"
         "mul.wide.u16    t4, %11, %21;\n\t"
         "add.cc.u32      r3, r3, t3;\n\t"
         "addc.cc.u32     r4, r4, t4;\n\t"
         "addc.u32        r5, r5, 0;\n\t"
         "mul.wide.u16    t2, %9, %19;\n\t"
         "mul.wide.u16    t3, %11, %19;\n\t"
         "mul.wide.u16    t4, %12, %20;\n\t"
         "mul.wide.u16    t5, %12, %22;\n\t"
         "add.cc.u32      r2, r2, t2;\n\t"
         "addc.cc.u32     r3, r3, t3;\n\t"
         "addc.cc.u32     r4, r4, t4;\n\t"
         "addc.cc.u32     r5, r5, t5;\n\t"
         "addc.u32        r6, r6, 0;\n\t"
         "mul.wide.u16    t2, %10, %18;\n\t"
         "mul.wide.u16    t3, %12, %18;\n\t"
         "mul.wide.u16    t4, %13, %19;\n\t"
         "mul.wide.u16    t5, %13, %21;\n\t"
         "add.cc.u32      r2, r2, t2;\n\t"
         "addc.cc.u32     r3, r3, t3;\n\t"
         "addc.cc.u32     r4, r4, t4;\n\t"
         "addc.cc.u32     r5, r5, t5;\n\t"
         "addc.u32        r6, r6, 0;\n\t"
         "mul.wide.u16    t1, %9, %17;\n\t"
         "mul.wide.u16    t2, %11, %17;\n\t"
         "mul.wide.u16    t3, %13, %17;\n\t"
         "mul.wide.u16    t4, %14, %18;\n\t"
         "mul.wide.u16    t5, %14, %20;\n\t"
         "mul.wide.u16    t6, %14, %22;\n\t"
         "add.cc.u32      r1, r1, t1;\n\t"
         "addc.cc.u32     r2, r2, t2;\n\t"
         "addc.cc.u32     r3, r3, t3;\n\t"
         "addc.cc.u32     r4, r4, t4;\n\t"
         "addc.cc.u32     r5, r5, t5;\n\t"
         "addc.cc.u32     r6, r6, t6;\n\t"
         "addc.u32        r7, r7, 0;\n\t"
         "mul.wide.u16    t1, %10, %16;\n\t"
         "mul.wide.u16    t2, %12, %16;\n\t"
         "mul.wide.u16    t3, %14, %16;\n\t"
         "mul.wide.u16    t4, %15, %17;\n\t"
         "mul.wide.u16    t5, %15, %19;\n\t"
         "mul.wide.u16    t6, %15, %21;\n\t"
         "add.cc.u32      r1, r1, t1;\n\t"
         "addc.cc.u32     r2, r2, t2;\n\t"
         "addc.cc.u32     r3, r3, t3;\n\t"
         "addc.cc.u32     r4, r4, t4;\n\t"
         "addc.cc.u32     r5, r5, t5;\n\t"
         "addc.cc.u32     r6, r6, t6;\n\t"
         "addc.u32        r7, r7, 0;\n\t"
         // compute second partial sum
         "mul.wide.u16    t0, %8, %17;\n\t"
         "mul.wide.u16    t1, %8, %19;\n\t"
         "mul.wide.u16    t2, %8, %21;\n\t"
         "mul.wide.u16    t3, %8, %23;\n\t"
         "mul.wide.u16    t4, %10, %23;\n\t"
         "mul.wide.u16    t5, %12, %23;\n\t"
         "mul.wide.u16    t6, %14, %23;\n\t"
         "mul.wide.u16    %3, %9, %22;\n\t"
         "add.cc.u32      t3, t3, %3;\n\t"
         "addc.u32        t4, t4, 0;\n\t"
         "mul.wide.u16    %3, %10, %21;\n\t"
         "add.cc.u32      t3, t3, %3;\n\t"
         "addc.u32        t4, t4, 0;\n\t"
         "mul.wide.u16    %2, %9, %20;\n\t"
         "mul.wide.u16    %3, %11, %20;\n\t"
         "mul.wide.u16    %4, %11, %22;\n\t"
         "add.cc.u32      t2, t2, %2;\n\t"
         "addc.cc.u32     t3, t3, %3;\n\t"
         "addc.cc.u32     t4, t4, %4;\n\t"
         "addc.u32        t5, t5, 0;\n\t"
         "mul.wide.u16    %2, %10, %19;\n\t"
         "mul.wide.u16    %3, %12, %19;\n\t"
         "mul.wide.u16    %4, %12, %21;\n\t"
         "add.cc.u32      t2, t2, %2;\n\t"
         "addc.cc.u32     t3, t3, %3;\n\t"
         "addc.cc.u32     t4, t4, %4;\n\t"
         "addc.u32        t5, t5, 0;\n\t"
         "mul.wide.u16    %1, %9, %18;\n\t"
         "mul.wide.u16    %2, %11, %18;\n\t"
         "mul.wide.u16    %3, %13, %18;\n\t"
         "mul.wide.u16    %4, %13, %20;\n\t"
         "mul.wide.u16    %5, %13, %22;\n\t"
         "add.cc.u32      t1, t1, %1;\n\t"
         "addc.cc.u32     t2, t2, %2;\n\t"
         "addc.cc.u32     t3, t3, %3;\n\t"
         "addc.cc.u32     t4, t4, %4;\n\t"
         "addc.cc.u32     t5, t5, %5;\n\t"
         "addc.u32        t6, t6, 0;\n\t"
         "mul.wide.u16    %1, %10, %17;\n\t"
         "mul.wide.u16    %2, %12, %17;\n\t"
         "mul.wide.u16    %3, %14, %17;\n\t"
         "mul.wide.u16    %4, %14, %19;\n\t"
         "mul.wide.u16    %5, %14, %21;\n\t"
         "add.cc.u32      t1, t1, %1;\n\t"
         "addc.cc.u32     t2, t2, %2;\n\t"
         "addc.cc.u32     t3, t3, %3;\n\t"
         "addc.cc.u32     t4, t4, %4;\n\t"
         "addc.cc.u32     t5, t5, %5;\n\t"
         "addc.u32        t6, t6, 0;\n\t"
         "mul.wide.u16    %0, %9, %16;\n\t"
         "mul.wide.u16    %1, %11, %16;\n\t"
         "mul.wide.u16    %2, %13, %16;\n\t"
         "mul.wide.u16    %3, %15, %16;\n\t"
         "mul.wide.u16    %4, %15, %18;\n\t"
         "mul.wide.u16    %5, %15, %20;\n\t"
         "mul.wide.u16    %6, %15, %22;\n\t"
         "add.cc.u32      t0, t0, %0;\n\t"
         "addc.cc.u32     t1, t1, %1;\n\t"
         "addc.cc.u32     t2, t2, %2;\n\t"
         "addc.cc.u32     t3, t3, %3;\n\t"
         "addc.cc.u32     t4, t4, %4;\n\t"
         "addc.cc.u32     t5, t5, %5;\n\t"
         "addc.cc.u32     t6, t6, %6;\n\t"
         "addc.u32        t7, 0, 0;\n\t"
         // offset second partial sum by 16 bits
         "shf.l.clamp.b32 %7, t6, t7, 16;\n\t"
         "shf.l.clamp.b32 %6, t5, t6, 16;\n\t"
         "shf.l.clamp.b32 %5, t4, t5, 16;\n\t"
         "shf.l.clamp.b32 %4, t3, t4, 16;\n\t"
         "shf.l.clamp.b32 %3, t2, t3, 16;\n\t"
         "shf.l.clamp.b32 %2, t1, t2, 16;\n\t"
         "shf.l.clamp.b32 %1, t0, t1, 16;\n\t"
         "shf.l.clamp.b32 %0,  0, t0, 16;\n\t"
         // add partial sums
         "add.cc.u32      %0, r0, %0;\n\t"
         "addc.cc.u32     %1, r1, %1;\n\t"
         "addc.cc.u32     %2, r2, %2;\n\t"
         "addc.cc.u32     %3, r3, %3;\n\t"
         "addc.cc.u32     %4, r4, %4;\n\t"
         "addc.cc.u32     %5, r5, %5;\n\t"
         "addc.cc.u32     %6, r6, %6;\n\t"
         "addc.u32        %7, r7, %7;\n\t"
         "}"
         : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3]), "=r"(c[4]), "=r"(c[5]), "=r"(c[6]), "=r"(c[7])
         : "h"(U[0]), "h"(U[1]), "h"(U[2]), "h"(U[3]), "h"(U[4]), "h"(U[5]), "h"(U[6]), "h"(U[7]),
           "h"(V[0]), "h"(V[1]), "h"(V[2]), "h"(V[3]), "h"(V[4]), "h"(V[5]), "h"(V[6]), "h"(V[7])
    );
}

__device__ __forceinline__ void sub_abs(uint32 *a, uint32 *v, const uint32 *b, const uint32 *c)
{
    asm ("{\n\t"
            ".reg .pred p1; \n\t"
            ".reg .u32 r0, r1, r2, r3; \n\t"
            ".reg .u32 t0; \n\t"

            "sub.cc.u32  r0, %10, %6; \n\t"
            "subc.cc.u32 r1, %11, %7; \n\t"
            "subc.cc.u32 r2, %12, %8; \n\t"
            "subc.u32    r3, %13, %9; \n\t"

            "sub.cc.u32  %0, %6, %10; \n\t"
            "subc.cc.u32 %1, %7, %11; \n\t"
            "subc.cc.u32 %2, %8, %12; \n\t"
            "subc.u32    %3, %9, %13; \n\t"

            "setp.gt.u32 p1, %9, %13; \n\t"
            "selp.b32 t0, 8, 0, p1; \n\t"
            "or.b32 %4, 0, t0; \n\t"

            "setp.gt.u32 p1, %8, %12; \n\t"
            "selp.b32 t0, 4, 0, p1; \n\t"
            "or.b32 %4, %4, t0; \n\t"

            "setp.gt.u32 p1, %7, %11; \n\t"
            "selp.b32 t0, 2, 0, p1; \n\t"
            "or.b32 %4, %4, t0; \n\t"

            "setp.gt.u32 p1, %6, %10; \n\t"
            "selp.b32 t0, 1, 0, p1; \n\t"
            "or.b32 %4, %4, t0; \n\t"

            "setp.lt.u32 p1, %9, %13; \n\t"
            "selp.b32 t0, 8, 0, p1; \n\t"
            "or.b32 %5, 0, t0; \n\t"

            "setp.lt.u32 p1, %8, %11; \n\t"
            "selp.b32 t0, 4, 0, p1; \n\t"
            "or.b32 %5, %5, t0; \n\t"

            "setp.lt.u32 p1, %7, %10; \n\t"
            "selp.b32 t0, 2, 0, p1; \n\t"
            "or.b32 %5, %5, t0; \n\t"

            "setp.lt.u32 p1, %6, %9; \n\t"
            "selp.b32 t0, 1, 0, p1; \n\t"
            "or.b32 %5, %5, t0; \n\t"

            "setp.lt.u32 p1, %4, %5; \n\t"

            "selp.b32    %0, r0, %0, p1; \n\t"
            "selp.b32    %1, r1, %1, p1; \n\t"
            "selp.b32    %2, r2, %2, p1; \n\t"
            "selp.b32    %3, r3, %3, p1; \n\t"
        "}"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]), "=r"(v[0]), "=r"(v[1])
        : "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}

// IF Y > Z return 1
__device__ __forceinline__ uint32 ge128(const uint32 *Y, const uint32 *Z)
{
    uint32 a = 0;

    asm ("{\n\t"
            ".reg .pred p1; \n\t"
            ".reg .u32 v0, v1; \n\t"
            ".reg .u32 t0; \n\t"

            "setp.gt.u32 p1, %3, %7; \n\t"
            "selp.b32 t0, 8, 0, p1; \n\t"
            "or.b32 v0, 0, t0; \n\t"

            "setp.gt.u32 p1, %2, %6; \n\t"
            "selp.b32 t0, 4, 0, p1; \n\t"
            "or.b32 v0, v0, t0; \n\t"

            "setp.gt.u32 p1, %1, %5; \n\t"
            "selp.b32 t0, 2, 0, p1; \n\t"
            "or.b32 v0, v0, t0; \n\t"

            "setp.gt.u32 p1, %0, %4; \n\t"
            "selp.b32 t0, 1, 0, p1; \n\t"
            "or.b32 v0, v0, t0; \n\t"

            "setp.lt.u32 p1, %3, %7; \n\t"
            "selp.b32 t0, 8, 0, p1; \n\t"
            "or.b32 v1, 0, t0; \n\t"

            "setp.lt.u32 p1, %2, %6; \n\t"
            "selp.b32 t0, 4, 0, p1; \n\t"
            "or.b32 v1, v1, t0; \n\t"

            "setp.lt.u32 p1, %1, %5; \n\t"
            "selp.b32 t0, 2, 0, p1; \n\t"
            "or.b32 v1, v1, t0; \n\t"

            "setp.lt.u32 p1, %0, %4; \n\t"
            "selp.b32 t0, 1, 0, p1; \n\t"
            "or.b32 v1, v1, t0; \n\t"

            "setp.lt.u32 p1, v0, v1; \n\t"
            "selp.u32 %0, 0, 1, p1; \n\t"
        "}"
        : "=r"(a)
        : "r"(Y[0]), "r"(Y[1]), "r"(Y[2]), "r"(Y[3]), "r"(Z[0]), "r"(Z[1]), "r"(Z[2]), "r"(Z[3])
    );

    return a;
}

__device__ __forceinline__ void mov_x2x16(uint16 *a, uint16 *d, const uint32 *b, const uint32 *c)
{
    asm ("{\n\t"
            "mov.b32 {%0, %1}, %16; \n\t"
            "mov.b32 {%2, %3}, %17; \n\t"
            "mov.b32 {%4, %5}, %18; \n\t"
            "mov.b32 {%6, %7}, %19; \n\t"
            "mov.b32 {%8, %9}, %20; \n\t"
            "mov.b32 {%10, %11}, %21; \n\t"
            "mov.b32 {%12, %13}, %22; \n\t"
            "mov.b32 {%14, %15}, %23; \n\t"
        "}"
        : "=h"(a[0]), "=h"(a[1]), "=h"(a[2]), "=h"(a[3]), "=h"(a[4]), "=h"(a[5]), "=h"(a[6]), "=h"(a[7]), "=h"(d[0]), "=h"(d[1]), "=h"(d[2]), "=h"(d[3]), "=h"(d[4]), "=h"(d[5]), "=h"(d[6]), "=h"(d[7])
        : "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}

__device__ __forceinline__ void neg_z(uint32 *z, uint32 *v1, uint32 *v2)
{
    asm ("{\n\t"
            ".reg .pred p1, p2, p3; \n\t"
            ".reg .u32 t0, t1, t2, t3, t4, t5, t6, t7; \n\t"
            ".reg .u32 r0; \n\t"

            "sub.cc.u32      t0, 0xffffffed, %0;  \n\t"
            "subc.cc.u32     t1, 0xffffffff, %1;  \n\t"
            "subc.cc.u32     t2, 0xffffffff, %2;  \n\t"
            "subc.cc.u32     t3, 0xffffffff, %3;  \n\t"
            "subc.cc.u32     t4, 0xffffffff, %4;  \n\t"
            "subc.cc.u32     t5, 0xffffffff, %5;  \n\t"
            "subc.cc.u32     t6, 0xffffffff, %6;  \n\t"
            "subc.cc.u32     t7, 0x7fffffff, %7;  \n\t"
            "subc.u32        r0,  0,  0; \n\t"
            "and.b32         r0, r0, 38; \n\t"
            "sub.cc.u32      t0, t0, r0; \n\t"
            "subc.cc.u32     t1, t1,  0; \n\t"
            "subc.cc.u32     t2, t2,  0; \n\t"
            "subc.cc.u32     t3, t3,  0; \n\t"
            "subc.cc.u32     t4, t4,  0; \n\t"
            "subc.cc.u32     t5, t5,  0; \n\t"
            "subc.cc.u32     t6, t6,  0; \n\t"
            "subc.u32        t7, t7,  0; \n\t"

            "setp.lt.u32 p1, %8, %9; \n\t"
            "setp.lt.u32 p2, %10, %11; \n\t"
            "xor.pred    p3, p1, p2; \n\t"

            "selp.b32    %0, t0, %0, p3; \n\t"
            "selp.b32    %1, t1, %1, p3; \n\t"
            "selp.b32    %2, t2, %2, p3; \n\t"
            "selp.b32    %3, t3, %3, p3; \n\t"
            "selp.b32    %4, t4, %4, p3; \n\t"
            "selp.b32    %5, t5, %5, p3; \n\t"
            "selp.b32    %6, t6, %6, p3; \n\t"
            "selp.b32    %7, t7, %7, p3; \n\t"
        "}"
        : "+r"(z[0]), "+r"(z[1]), "+r"(z[2]), "+r"(z[3]), "+r"(z[4]), "+r"(z[5]), "+r"(z[6]), "+r"(z[7])
        : "r"(v1[0]), "r"(v1[1]), "r"(v2[0]), "r"(v2[1])
    );
}

/* Computes Z = X*Y mod P. */
/* Output fits into 8 words but could be greater than P */
__device__ __forceinline__ void mul_reduce_5(uint32 *Z, const uint32 *E, const uint32 *H)
{
    uint32 v_pred1[2];
    uint32 v_pred2[2];
    
    uint32 z1[8], X[8], Y[8];
    t128 U, V;

    sub_abs(U.u32, v_pred1, E+4, E); // U = y1-y0
    sub_abs(V.u32, v_pred2, H, H+4); // V = x0-x1
    umul128wide(z1, U.u16, V.u16); // z1 = U*V

    mov_x2x16(U.u16, V.u16, H, E);
    umul128wide(X, U.u16, V.u16); // X = x0*y0

    neg_z(z1, v_pred1, v_pred2);

    add_reduce(z1, X, z1); // z1 = X + z1

    mov_x2x16(U.u16, V.u16, H+4, E+4);
    umul128wide(Y, U.u16, V.u16);  // Y = x1*y1

    add_reduce(z1, Y, z1);  // z1 = Y + z1

    add_reduce_1(X, z1,   X);  
    add_reduce_2(Y, z1+4, Y);

    asm("{                                        \n\t"
        "mad.lo.cc.u32  %0, 38, %8, %16;          \n\t"
        "madc.hi.u32    %16, 38, %8, 0;           \n\t"
        "mad.lo.cc.u32  %1, 38, %9, %16;          \n\t"
        "madc.hi.u32    %16, 38, %9, 0;           \n\t"
        "add.cc.u32     %1, %1, %17;              \n\t"
        "madc.lo.cc.u32  %2, 38, %10, %16;         \n\t"
        "madc.hi.u32    %16, 38, %10, 0;          \n\t"
        "add.cc.u32     %2, %2, %18;              \n\t"
        "madc.lo.cc.u32  %3, 38, %11, %16;         \n\t"
        "madc.hi.u32    %16, 38, %11, 0;          \n\t"
        "add.cc.u32     %3, %3, %19;              \n\t"
        "madc.lo.cc.u32  %4, 38, %12, %16;         \n\t"
        "madc.hi.u32    %16, 38, %12, 0;          \n\t"
        "add.cc.u32     %4, %4, %20;              \n\t"
        "madc.lo.cc.u32  %5, 38, %13, %16;         \n\t"
        "madc.hi.u32    %16, 38, %13, 0;          \n\t"
        "add.cc.u32     %5, %5, %21;              \n\t"
        "madc.lo.cc.u32  %6, 38, %14, %16;         \n\t"
        "madc.hi.u32    %16, 38, %14, 0;          \n\t"
        "add.cc.u32     %6, %6, %22;              \n\t"
        "madc.lo.cc.u32  %7, 38, %15, %16;         \n\t"
        "madc.hi.u32    %16, 38, %15, 0;          \n\t"
        "add.cc.u32     %7, %7, %23;              \n\t"
        "addc.u32       %16, %16, 0;              \n\t"
        "mul.lo.u32     %16, %16, 38;             \n\t"
        "add.cc.u32     %0, %0, %16;              \n\t"
        "addc.u32       %1, %1,  0;               \n\t"
        "}"
        // %0 - %7
        : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3]), "=r"(Z[4]), "=r"(Z[5]), "=r"(Z[6]), "=r"(Z[7])
        // %8 - %15
        : "r"(Y[0]), "r"(Y[1]), "r"(Y[2]), "r"(Y[3]), "r"(Y[4]), "r"(Y[5]), "r"(Y[6]), "r"(Y[7]),
        // %16 - %23
          "r"(X[0]), "r"(X[1]), "r"(X[2]), "r"(X[3]), "r"(X[4]), "r"(X[5]), "r"(X[6]), "r"(X[7])
        );
}

/* Computes Z = X*Y mod P. */
/* Output fits into 8 words but could be greater than P */
__device__ __forceinline__ void mul_reduce_2(uint32 *Z, const uint32 *E, const uint32 *H)
{
    uint32 v_pred1[2];
    uint32 v_pred2[2];
    
    uint32 z1[8], X[8], Y[8];
    t128 U, V;

    sub_abs(U.u32, v_pred1, E+4, E); // U = y1-y0
    sub_abs(V.u32, v_pred2, H, H+4); // V = x0-x1
    umul128wide(z1, U.u16, V.u16); // z1 = U*V

    mov_x2x16(U.u16, V.u16, H, E);
    umul128wide(X, U.u16, V.u16); // X = x0*y0

    neg_z(z1, v_pred1, v_pred2);

    add_reduce(z1, X, z1); // z1 = X + z1

    mov_x2x16(U.u16, V.u16, H+4, E+4);
    umul128wide(Y, U.u16, V.u16);  // Y = x1*y1

    add_reduce(z1, Y, z1);  // z1 = Y + z1

    add_reduce_1(X, z1,   X);  
    add_reduce_2(Y, z1+4, Y);

    asm("{                                        \n\t"
        "mad.lo.cc.u32  %0, 38, %8, %16;          \n\t"
        "madc.hi.u32    %16, 38, %8, 0;           \n\t"
        "mad.lo.cc.u32  %1, 38, %9, %16;          \n\t"
        "madc.hi.u32    %16, 38, %9, 0;           \n\t"
        "add.cc.u32     %1, %1, %17;              \n\t"
        "madc.lo.cc.u32  %2, 38, %10, %16;         \n\t"
        "madc.hi.u32    %16, 38, %10, 0;          \n\t"
        "add.cc.u32     %2, %2, %18;              \n\t"
        "madc.lo.cc.u32  %3, 38, %11, %16;         \n\t"
        "madc.hi.u32    %16, 38, %11, 0;          \n\t"
        "add.cc.u32     %3, %3, %19;              \n\t"
        "madc.lo.cc.u32  %4, 38, %12, %16;         \n\t"
        "madc.hi.u32    %16, 38, %12, 0;          \n\t"
        "add.cc.u32     %4, %4, %20;              \n\t"
        "madc.lo.cc.u32  %5, 38, %13, %16;         \n\t"
        "madc.hi.u32    %16, 38, %13, 0;          \n\t"
        "add.cc.u32     %5, %5, %21;              \n\t"
        "madc.lo.cc.u32  %6, 38, %14, %16;         \n\t"
        "madc.hi.u32    %16, 38, %14, 0;          \n\t"
        "add.cc.u32     %6, %6, %22;              \n\t"
        "madc.lo.cc.u32  %7, 38, %15, %16;         \n\t"
        "madc.hi.u32    %16, 38, %15, 0;          \n\t"
        "add.cc.u32     %7, %7, %23;              \n\t"
        "addc.u32       %16, %16, 0;              \n\t"
        "mul.lo.u32     %16, %16, 38;             \n\t"
        "add.cc.u32     %0, %0, %16;              \n\t"
        "addc.u32       %1, %1,  0;               \n\t"
        "}"
        // %0 - %7
        : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3]), "=r"(Z[4]), "=r"(Z[5]), "=r"(Z[6]), "=r"(Z[7])
        // %8 - %15
        : "r"(Y[0]), "r"(Y[1]), "r"(Y[2]), "r"(Y[3]), "r"(Y[4]), "r"(Y[5]), "r"(Y[6]), "r"(Y[7]),
        // %16 - %23
          "r"(X[0]), "r"(X[1]), "r"(X[2]), "r"(X[3]), "r"(X[4]), "r"(X[5]), "r"(X[6]), "r"(X[7])
        );
}

/* Computes Z = X*Y mod P. */
__device__ __forceinline__ void mul_reduce_I(uint32 *Z, const uint32 *H)
{
    uint32 z2[4], z1[8], X[8], Y[8], neg_z1[8];
    t128 U, V;
    uint32 v_pred1[2];

    uint32 z0[4] = {0x0C12C909, 0x99A11A8E, 0x5D6E056D, 0x03BFF386};
    sub_abs(z2, v_pred1, H, H+4);
    mov_x2x16(U.u16, V.u16, z2, z0);
    umul128wide(z1, U.u16, V.u16);

    neg256_mod(neg_z1, z1);

    #define bn_neg_set2(i)                \
        z1[i] = (v_pred1[0] > v_pred1[1]) ? neg_z1[i] : z1[i];

    unroll_8(bn_neg_set2);

    uint32 E0[4] = {0x4a0ea0b0, 0xc4ee1b27, 0xad2fe478, 0x2f431806};
    mov_x2x16(U.u16, V.u16, H, E0);
    umul128wide(X, U.u16, V.u16);
    add_reduce(z1, X, z1);

    uint32 E1[4] = {0x3dfbd7a7, 0x2b4d0099, 0x4fc1df0b, 0x2b832480};
    mov_x2x16(U.u16, V.u16, H+4, E1);
    umul128wide(Y, U.u16, V.u16);
    add_reduce(z1, Y, z1);
   
    add_reduce_1(X, z1,   X);
    add_reduce_2(Y, z1+4, Y);

    asm("{                                        \n\t"
        "mad.lo.cc.u32  %0, 38, %8, %16;          \n\t"
        "madc.hi.u32    %16, 38, %8, 0;           \n\t"
        "mad.lo.cc.u32  %1, 38, %9, %16;          \n\t"
        "madc.hi.u32    %16, 38, %9, 0;           \n\t"
        "add.cc.u32     %1, %1, %17;              \n\t"
        "madc.lo.cc.u32  %2, 38, %10, %16;         \n\t"
        "madc.hi.u32    %16, 38, %10, 0;          \n\t"
        "add.cc.u32     %2, %2, %18;              \n\t"
        "madc.lo.cc.u32  %3, 38, %11, %16;         \n\t"
        "madc.hi.u32    %16, 38, %11, 0;          \n\t"
        "add.cc.u32     %3, %3, %19;              \n\t"
        "madc.lo.cc.u32  %4, 38, %12, %16;         \n\t"
        "madc.hi.u32    %16, 38, %12, 0;          \n\t"
        "add.cc.u32     %4, %4, %20;              \n\t"
        "madc.lo.cc.u32  %5, 38, %13, %16;         \n\t"
        "madc.hi.u32    %16, 38, %13, 0;          \n\t"
        "add.cc.u32     %5, %5, %21;              \n\t"
        "madc.lo.cc.u32  %6, 38, %14, %16;         \n\t"
        "madc.hi.u32    %16, 38, %14, 0;          \n\t"
        "add.cc.u32     %6, %6, %22;              \n\t"
        "madc.lo.cc.u32  %7, 38, %15, %16;         \n\t"
        "madc.hi.u32    %16, 38, %15, 0;          \n\t"
        "add.cc.u32     %7, %7, %23;              \n\t"
        "addc.u32       %16, %16, 0;              \n\t"
        "mul.lo.u32     %16, %16, 38;             \n\t"
        "add.cc.u32     %0, %0, %16;              \n\t"
        "addc.u32       %1, %1,  0;               \n\t"
        "}"
        // %0 - %7
        : "=r"(Z[0]), "=r"(Z[1]), "=r"(Z[2]), "=r"(Z[3]), "=r"(Z[4]), "=r"(Z[5]), "=r"(Z[6]), "=r"(Z[7])
        // %8 - %15
        : "r"(Y[0]), "r"(Y[1]), "r"(Y[2]), "r"(Y[3]), "r"(Y[4]), "r"(Y[5]), "r"(Y[6]), "r"(Y[7]),
        // %16 - %23
          "r"(X[0]), "r"(X[1]), "r"(X[2]), "r"(X[3]), "r"(X[4]), "r"(X[5]), "r"(X[6]), "r"(X[7])
        );
}

/* Computes Z = X*Y mod P. */
__device__ __forceinline__ void mul_mod_2(uint32* Z, const uint32* X, const uint32* Y)
{
    mul_reduce_2(Z, X, Y);
    mod_p(Z);
}

/* Computes Z = X*Y mod P. */
__device__ __forceinline__ void mul_mod_I(uint32* Z, const uint32* Y)
{
    mul_reduce_I(Z, Y);
    mod_p(Z);
}


__global__ void test_mul256(t256 *aa, t256 *b, t256 *c)
{
    t256 x, y;

	const int gid = blockDim.x * blockIdx.x + threadIdx.x;

    copy(y.u32, aa[gid].u32);
    copy(x.u32, b[gid].u32);

    mul_mod_I(x.u32, y.u32);

    copy(c[gid].u32, x.u32);
}