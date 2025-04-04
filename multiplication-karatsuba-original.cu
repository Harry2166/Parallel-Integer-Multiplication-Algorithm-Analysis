#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<cuda_runtime.h>

// the following functions are from: https://github.com/Rakshithkumar26/CUDA-Code-for-Karatsuba-Algorithm/ 
__device__ int numDigits(long long n) {
	int count = 0;
	while (n != 0) {
		n /= 10;
		count++;
	}
	return count;
}

__device__ int customMax(int a, int b) {
    return (a > b) ? a : b;
}

__global__ void multiplication(long long *d_a, long long *d_b, long long *d_c, int len) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<len){
      int x=d_a[tid];
      int y=d_b[tid];
      if (x < 10 || y < 10) {
			  d_c[tid] = x * y;
		  } else {
			  // Calculate the number of digits in the two numbers and divide by 2
			  int n = customMax(numDigits(x), numDigits(y));
			  int n2 = (n / 2);

			  // Split the numbers into two parts

        long long x_h = x / (long long)pow(10, n2);
        long long x_l = x % (long long)pow(10, n2);
        long long y_h = y / (long long)pow(10, n2);
        long long y_l = y % (long long)pow(10, n2);

        // Recursively calculate the three products
        long long high_prod = x_h * y_h;
        long long low_prod = x_l * y_l;
        long long inter_prod = ((x_h + x_l) * (y_h + y_l));
        long long subtract = inter_prod - high_prod - low_prod;

        // Calculate and return the final result
        d_c[tid] = (high_prod * (long long)pow(10, 2 * n2)) + (subtract * (long long)pow(10, n2)) + low_prod;
      }
    }
}
