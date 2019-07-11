#ifndef MAIN_H_
#define MAIN_H_

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include "common.cuh"
#include "macros.h"

__device__ int gcd(int a, int b);
__device__ int lcm(const int a, const int b, const int gcd);
__device__ int f(const int a, const int b, const int minLcm);
__global__ void buddyKVGSum(const Matrix a,
    const Matrix b,
    Matrix c,
    const int n,
    const int min_lcm);

#endif /* MAIN_H_ */