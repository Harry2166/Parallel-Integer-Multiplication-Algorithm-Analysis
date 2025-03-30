
#include <stdio.h>
#include <cuda.h>

#define DIGITS 10 
#define BLOCK_WIDTH 16 

__global__ void bmulTiled(size_t *Aglb, size_t *Bglb, uint64_t *Cglb) {
    __shared__ size_t Ash[BLOCK_WIDTH], Bsh[BLOCK_WIDTH]; 
    __shared__ uint64_t Csh[2 * BLOCK_WIDTH];

    int ii = blockIdx.y * BLOCK_WIDTH, i = threadIdx.y; // 0 <= i < T 
    int jj = blockIdx.x * BLOCK_WIDTH, j = threadIdx.x; // 0 <= j < T
    
    // copy A and B from global to shared memory & initialize Csh
    if (threadIdx.y == 0) {
        Ash[j] = Aglb[ii + j]; 
        Bsh[j] = Bglb[jj + j]; 
        Csh[j] = 0;
        Csh[j + BLOCK_WIDTH] = 0;
    }
    __syncthreads();

    if (ii + jj + i + j < DIGITS) {
        uint64_t prod = ((uint64_t)Ash[i]) * ((uint64_t)Bsh[j]);
        atomicAdd(&Csh[i + j], prod); // atomic in shared memory
    }
    __syncthreads();

    int tid = i * BLOCK_WIDTH + j;
    if (tid < 2 * BLOCK_WIDTH && ii + jj + tid < 2 * DIGITS) {  // atomic in global memory 
      atomicAdd(&Cglb[ii + jj + tid], Csh[tid]); 
    }
}

__global__ void carryPropagation(uint64_t *Cglb, size_t *Result, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    uint64_t carry = 0;
    for (int i = 0; i < size; i++) {
        uint64_t temp = Cglb[i] + carry;
        Result[i] = temp % 10; 
        carry = temp / 10;      
    }

    if (carry > 0 && size < 2 * DIGITS) { 
        Result[size] = carry;
    }
}

void printArray(size_t *arr, int size) {
    bool leadingZero = true;
    for (int i = size - 1; i >= 0; i--) {
        if (arr[i] != 0) leadingZero = false;
        if (!leadingZero) printf("%llu", arr[i]);
    }
    if (leadingZero) printf("0");
    printf("\n");
}

int main() {
    printf("Tiled Quadratic Multiplication: ");
    size_t h_A[DIGITS] = {5, 4, 3, 2, 1}; // 12345 in reverse
    size_t h_B[DIGITS] = {6, 5, 4, 3, 2}; // 23456 in reverse
    size_t h_Result[2 * DIGITS] = {0};  

    size_t *d_A, *d_B;
    uint64_t *d_C;
    size_t *d_Result;

    cudaMalloc((void **)&d_A, DIGITS * sizeof(size_t));
    cudaMalloc((void **)&d_B, DIGITS * sizeof(size_t));
    cudaMalloc((void **)&d_C, 2 * DIGITS * sizeof(uint64_t));
    cudaMalloc((void **)&d_Result, 2 * DIGITS * sizeof(size_t));

    cudaMemcpy(d_A, h_A, DIGITS * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DIGITS * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, 2 * DIGITS * sizeof(uint64_t));
    cudaMemset(d_Result, 0, 2 * DIGITS * sizeof(size_t));

    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 dimGrid((DIGITS + BLOCK_WIDTH) / BLOCK_WIDTH, (DIGITS + BLOCK_WIDTH - 1) / BLOCK_WIDTH);

    printArray(h_A, DIGITS);
    printArray(h_B, DIGITS);

    bmulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    carryPropagation<<<(DIGITS + 255) / 256, 256>>>(d_C, d_Result, 2 * DIGITS);
    cudaMemcpy(h_Result, d_Result, 2 * DIGITS * sizeof(size_t), cudaMemcpyDeviceToHost);

    printArray(h_Result, 2 * DIGITS);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_Result);

    return 0;
  }
