
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void karatsuba(uint64_t *X, uint64_t *Y, uint64_t *result) {
    uint64_t x = *X;
    uint64_t y = *Y;

    printf("x = %llu; y = %llu\n", x, y);

    if (x == 0 || y == 0){
      *result = 0;
      return;
    }

    if (x < 2 && y < 2) {
        *result = x & y;
        return;
    }

    int bitsX = __log2f((float)x) + 1;
    int bitsY = __log2f((float)y) + 1;

    int maxBits = max(bitsX, bitsY);
    int mid = maxBits >> 1;

    // Splitting the numbers
    uint64_t XH = x >> mid;
    uint64_t XL = x & ((1 << mid) - 1);
    uint64_t YH = y >> mid;
    uint64_t YL = y & ((1 << mid) - 1);

    uint64_t *A, *B, *C;
    cudaMalloc(&A, sizeof(uint64_t));
    cudaMalloc(&B, sizeof(uint64_t));
    cudaMalloc(&C, sizeof(uint64_t));

    uint64_t *XH_d, *YH_d, *XL_d, *YL_d;
    cudaMalloc(&XH_d, sizeof(uint64_t));
    cudaMalloc(&YH_d, sizeof(uint64_t));
    cudaMalloc(&XL_d, sizeof(uint64_t));
    cudaMalloc(&YL_d, sizeof(uint64_t));

    *XH_d = XH;
    *YH_d = YH;
    *XL_d = XL;
    *YL_d = YL;

    karatsuba<<<1, 1>>>(XH_d, YH_d, A);  // A = XH * YH
    karatsuba<<<1, 1>>>(XL_d, YL_d, B);  // B = XL * YL

    uint64_t XH_plus_XL = XH + XL;
    uint64_t YH_plus_YL = YH + YL;

    uint64_t *XH_plus_XL_d, *YH_plus_YL_d;
    cudaMalloc(&XH_plus_XL_d, sizeof(uint64_t));
    cudaMalloc(&YH_plus_YL_d, sizeof(uint64_t));

    *XH_plus_XL_d = XH_plus_XL;
    *YH_plus_YL_d = YH_plus_YL;

    karatsuba<<<1, 1>>>(XH_plus_XL_d, YH_plus_YL_d, C);  // C = (XH + XL) * (YH + YL)

    uint64_t resA = *A; 
    uint64_t resB = *B;
    uint64_t resC = *C;

    uint64_t D = resC - resA - resB;
    *result = (resA << (mid << 1)) + (D << mid) + resB;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(XH_d);
    cudaFree(YH_d);
    cudaFree(XL_d);
    cudaFree(YL_d);
    cudaFree(XH_plus_XL_d);
    cudaFree(YH_plus_YL_d);
}

int main() {
    printf("Parallel Karatsuba from Kumar: \n");
    uint64_t X = 3;
    uint64_t Y = 2;
    uint64_t result = 0;

    uint64_t *X_d, *Y_d, *result_d;

    cudaMalloc((void**)&X_d, sizeof(uint64_t));
    cudaMalloc((void**)&Y_d, sizeof(uint64_t));
    cudaMalloc((void**)&result_d, sizeof(uint64_t));

    cudaMemcpy(X_d, &X, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(Y_d, &Y, sizeof(uint64_t), cudaMemcpyHostToDevice);

    karatsuba<<<1, 1>>>(X_d, Y_d, result_d);

    cudaDeviceSynchronize(); 

    cudaMemcpy(&result, result_d, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    printf("Karatsuba Result: %llu\n", result);

    cudaFree(X_d);
    cudaFree(Y_d);
    cudaFree(result_d);

    return 0;
}
